import React, {useMemo, useCallback, useEffect, useRef} from 'react';
import './EditorBottomNavigationBar.scss';
import {ImageData, LabelName, Subject} from '../../../store/labels/types';
import {AppState} from '../../../store';
import {connect} from 'react-redux';
import {ISize} from '../../../interfaces/ISize';
import {ContextType} from '../../../data/enums/ContextType';
import classNames from 'classnames';
import {TimeUtil} from '../../../utils/TimeUtil';
import CustomTimeline from './CustomTimeline/CustomTimeline';
import {TimelineTrack} from './CustomTimeline/types';
import {updateActiveLabelId, updateActiveSubjectId} from '../../../store/labels/actionCreators';
import {DEFAULT_SUBJECT_ID} from '../../../store/labels/reducer';
import {PlayheadClock} from './playheadClock';
import {LabelActions} from '../../../logic/actions/LabelActions';
import Tooltip from '../../Common/Tooltip/Tooltip';

interface IProps {
    size: ISize;
    activeContext: ContextType;
    currentTime: number;
    duration: number;
    isPlaying: boolean;
    frameRate: number;
    onStepSeconds: (direction: 'forward' | 'backward') => void;
    onJumpBoundary: (direction: 'forward' | 'backward') => void;
    onJumpToTime: (time: string) => void;
    imageData: ImageData;
    labelNames: LabelName[];
    subjects: Subject[];
    activeSubjectId: string | null;
    zoom: number;
    activeLabelId: string;
    focusedLabelNameId: string | null;
    updateActiveLabelIdAction: (activeLabelId: string) => any;
    updateActiveSubjectIdAction: (id: string | null) => any;
    playheadClock?: PlayheadClock;
}

const EditorBottomNavigationBar: React.FC<IProps> = ({
    size,
    activeContext,
    currentTime,
    duration,
    isPlaying,
    frameRate,
    onStepSeconds,
    onJumpBoundary,
    onJumpToTime,
    imageData,
    labelNames,
    subjects,
    activeSubjectId,
    zoom,
    activeLabelId,
    focusedLabelNameId,
    updateActiveLabelIdAction,
    updateActiveSubjectIdAction,
    playheadClock
}) => {
    const isCompactLayout = (size?.width || 0) < 1080 || zoom >= 2.4;
    const labelColumnWidth = isCompactLayout ? 112 : 132;
    const timeInfoWidth = isCompactLayout ? 172 : 196;
    const scaleWidth = Math.max(
        80,
        Math.min(
            160,
            (isCompactLayout ? 80 : 90) + Math.max(0, zoom - 1) * 24
        )
    );

    const fallbackSubjectId = subjects[0]?.id ?? DEFAULT_SUBJECT_ID;

    const tracks: TimelineTrack[] = useMemo(() => {
        const rects = imageData.labelRects || [];
        const labelNameById = new Map(labelNames.map((label) => [label.id, label]));
        const subjectIdSet = new Set(subjects.map(s => s.id));
        const clipsBySubjectId = new Map<string, TimelineTrack['clips']>();

        for (const subj of subjects) {
            clipsBySubjectId.set(subj.id, []);
        }

        for (const rect of rects) {
            if (!rect.labelId || (!rect.timestamp && typeof rect.frame !== 'number')) {
                continue;
            }

            const label = labelNameById.get(rect.labelId);
            if (!label) {
                continue;
            }

            // Focus mode: hide every clip that doesn't belong to the
            // focused behavior so the user can study one category at a
            // time. Clipping to `focusedLabelNameId` here keeps
            // downstream measurements (clipCount, selectedClip lookups,
            // boundary navigation via the same `tracks` data) consistent
            // with what the user sees.
            if (focusedLabelNameId && label.id !== focusedLabelNameId) {
                continue;
            }

            // Legacy clips (pre-multi-animal) lack animalId; route them to the
            // first subject so they render rather than getting dropped.
            const rawAnimalId = rect.animalId ?? fallbackSubjectId;
            const subjectId = subjectIdSet.has(rawAnimalId) ? rawAnimalId : fallbackSubjectId;

            let start: number;
            if (rect.timestamp) {
                start = TimeUtil.parseTimestamp(rect.timestamp);
            } else {
                start = rect.frame! / frameRate;
            }

            const hasEndTime = !!rect.endTimestamp || typeof rect.endFrame === 'number';
            // Ongoing clips still need a non-zero default while recording so they render.
            let end = start + 1 / frameRate;

            if (rect.endTimestamp) {
                end = TimeUtil.parseTimestamp(rect.endTimestamp);
            } else if (typeof rect.endFrame === 'number') {
                end = rect.endFrame / frameRate;
            }

            clipsBySubjectId.get(subjectId)?.push({
                id: rect.id,
                start,
                // Don't pad the visual span beyond the stored end time — short
                // clips (even single-frame ones) should render at their true
                // width. Clickability is handled by MIN_HIT_WIDTH in the timeline.
                end: Math.max(end, start),
                labelId: label.id,
                color: label.color || '#5297FF',
                labelName: label.name,
                isOngoing: !hasEndTime,
                animalId: subjectId
            });
        }

        return subjects.map((subj) => ({
            id: subj.id,
            name: subj.name,
            color: '#5297FF',
            clips: clipsBySubjectId.get(subj.id) || []
        }));
    }, [imageData.labelRects, labelNames, frameRate, subjects, fallbackSubjectId, focusedLabelNameId]);

    const clipCount = useMemo(() => {
        return tracks.reduce((sum, t) => sum + t.clips.length, 0);
    }, [tracks]);

    const selectedClip = useMemo(() => {
        if (!activeLabelId) return null;
        for (const track of tracks) {
            const found = track.clips.find((clip) => clip.id === activeLabelId);
            if (found) return found;
        }
        return null;
    }, [tracks, activeLabelId]);

    // When nothing is explicitly selected, fall back to the clip currently
    // under the playhead so the details chip surfaces during playback. The
    // active subject's track is checked first so overlapping clips on other
    // subjects don't shadow the one the user is actually watching.
    const playheadClip = useMemo(() => {
        if (selectedClip) return null;
        const eps = 1e-6;
        const orderedTracks = [
            ...tracks.filter((t) => t.id === activeSubjectId),
            ...tracks.filter((t) => t.id !== activeSubjectId)
        ];
        for (const track of orderedTracks) {
            for (const clip of track.clips) {
                const end = clip.isOngoing ? Math.max(currentTime, clip.end) : clip.end;
                if (currentTime + eps >= clip.start && currentTime - eps <= end) {
                    return clip;
                }
            }
        }
        return null;
    }, [tracks, currentTime, selectedClip, activeSubjectId]);

    const displayClip = selectedClip ?? playheadClip;

    const rowLabelsRef = useRef<HTMLDivElement>(null);

    const onStepSecondsBack = useCallback(() => onStepSeconds('backward'), [onStepSeconds]);
    const onStepSecondsForward = useCallback(() => onStepSeconds('forward'), [onStepSeconds]);
    const onJumpBoundaryBack = useCallback(() => onJumpBoundary('backward'), [onJumpBoundary]);
    const onJumpBoundaryForward = useCallback(() => onJumpBoundary('forward'), [onJumpBoundary]);
    const preventTransportButtonFocus = useCallback((event: React.MouseEvent<HTMLButtonElement>) => {
        event.preventDefault();
    }, []);
    const blurTransportButton = useCallback((event: React.FocusEvent<HTMLButtonElement>) => {
        event.currentTarget.blur();
    }, []);
    const handleTransportButtonClick = useCallback((
        event: React.MouseEvent<HTMLButtonElement>,
        action: () => void
    ) => {
        action();
        event.currentTarget.blur();
    }, []);

    const prevBoundaryTooltip = (
        <div className='BoundaryTooltip'>
            <div className='BoundaryTooltipTitle'>Jump to previous behavior boundary</div>
            <div className='BoundaryTooltipBody'>
                Snaps the playhead to the nearest clip start or end before the current time.
            </div>
            <div className='BoundaryTooltipShortcut'>
                Shortcut: <kbd>Shift</kbd> + <kbd>←</kbd>
            </div>
        </div>
    );

    const nextBoundaryTooltip = (
        <div className='BoundaryTooltip'>
            <div className='BoundaryTooltipTitle'>Jump to next behavior boundary</div>
            <div className='BoundaryTooltipBody'>
                Snaps the playhead to the nearest clip start or end after the current time.
            </div>
            <div className='BoundaryTooltipShortcut'>
                Shortcut: <kbd>Shift</kbd> + <kbd>→</kbd>
            </div>
        </div>
    );

    const onSelectClip = useCallback((clipId: string | null) => {
        updateActiveLabelIdAction(clipId);
        if (clipId) {
            for (const track of tracks) {
                const found = track.clips.find((clip) => clip.id === clipId);
                if (found && found.animalId && found.animalId !== activeSubjectId) {
                    updateActiveSubjectIdAction(found.animalId);
                }
                if (found) break;
            }
        }
    }, [updateActiveLabelIdAction, updateActiveSubjectIdAction, tracks, activeSubjectId]);

    useEffect(() => {
        if (!activeLabelId) return undefined;
        const handleDocMouseDown = (e: MouseEvent) => {
            const target = e.target as HTMLElement | null;
            if (!target) return;
            // Keep the selection alive when clicking on anything that should
            // either preserve it (the selection chip) or (re)select a clip
            // (a timeline clip or a row in the side list).
            if (target.closest('.ClipBar, .ClipRow, .SelectionChip')) return;
            updateActiveLabelIdAction(null);
        };
        document.addEventListener('mousedown', handleDocMouseDown);
        return () => document.removeEventListener('mousedown', handleDocMouseDown);
    }, [activeLabelId, updateActiveLabelIdAction]);

    const summaryItems = useMemo(() => {
        return [{label: 'Behavior clips', value: String(clipCount)}];
    }, [clipCount]);

    return (
        <div
            className={classNames(
                'EditorBottomNavigationBar',
                {
                    compact: isCompactLayout,
                    'with-context': activeContext === ContextType.EDITOR
                }
            )}
            style={{
                ['--timeline-label-column-width' as string]: `${labelColumnWidth}px`,
                ['--timeline-info-width' as string]: `${timeInfoWidth}px`
            }}
        >
            <div className='TimelineSidebar'>
                <div className='TimelineSidebarHeader'>
                    <span className='TimelineEyebrow'>Subjects</span>
                    <span className='TimelineSubtitle'>Press 1–9 to switch</span>
                </div>
                <div className='RowLabels' ref={rowLabelsRef}>
                    {subjects.length === 0 && (
                        <div className='RowLabelsEmpty'>No subjects</div>
                    )}
                    {subjects.map((s, i) => {
                        const isActive = s.id === activeSubjectId;
                        return (
                            <button
                                type='button'
                                key={s.id}
                                className={classNames('RowLabel', 'SubjectRowLabel', {active: isActive})}
                                onClick={() => updateActiveSubjectIdAction(s.id)}
                                title={`Switch to ${s.name} (key ${i + 1})`}
                            >
                                <span className='SubjectRowDigit'>{i + 1}</span>
                                <span className='SubjectRowName'>{s.name}</span>
                                {isActive && <span className='SubjectRowActiveDot' aria-hidden='true' />}
                            </button>
                        );
                    })}
                </div>
            </div>
            <div className='TimelineMain'>
                <div className='TimelineHeader'>
                    <div className='TransportControls'>
                        <button
                            type='button'
                            className='TransportButton'
                            tabIndex={-1}
                            onMouseDown={preventTransportButtonFocus}
                            onFocus={blurTransportButton}
                            onClick={(event) => handleTransportButtonClick(event, onStepSecondsBack)}
                            aria-label='Back ten seconds'
                        >
                            -10s
                        </button>
                        <Tooltip text={prevBoundaryTooltip} placement='top' maxWidth={260}>
                            <button
                                type='button'
                                className='TransportButton TransportButtonBoundary'
                                tabIndex={-1}
                                onMouseDown={preventTransportButtonFocus}
                                onFocus={blurTransportButton}
                                onClick={(event) => handleTransportButtonClick(event, onJumpBoundaryBack)}
                                aria-label='Jump to previous behavior boundary'
                            >
                                ⇤
                            </button>
                        </Tooltip>
                        <Tooltip text={nextBoundaryTooltip} placement='top' maxWidth={260}>
                            <button
                                type='button'
                                className='TransportButton TransportButtonBoundary'
                                tabIndex={-1}
                                onMouseDown={preventTransportButtonFocus}
                                onFocus={blurTransportButton}
                                onClick={(event) => handleTransportButtonClick(event, onJumpBoundaryForward)}
                                aria-label='Jump to next behavior boundary'
                            >
                                ⇥
                            </button>
                        </Tooltip>
                        <button
                            type='button'
                            className='TransportButton'
                            tabIndex={-1}
                            onMouseDown={preventTransportButtonFocus}
                            onFocus={blurTransportButton}
                            onClick={(event) => handleTransportButtonClick(event, onStepSecondsForward)}
                            aria-label='Forward ten seconds'
                        >
                            +10s
                        </button>
                    </div>
                    {displayClip && (
                        <div className='SelectionChip' title={displayClip.labelName}>
                            <span className='SelectionChipDot' style={{backgroundColor: displayClip.color}} />
                            <span className='SelectionChipName'>{displayClip.labelName}</span>
                            <span className='SelectionChipSep' aria-hidden='true' />
                            <span className='SelectionChipTime'>
                                {TimeUtil.formatTime(displayClip.start)}
                                <span className='SelectionChipArrow' aria-hidden='true'>→</span>
                                {displayClip.isOngoing ? '—' : TimeUtil.formatTime(displayClip.end)}
                            </span>
                            <span className='SelectionChipSep' aria-hidden='true' />
                            <span className='SelectionChipDuration'>
                                {Math.max(displayClip.end - displayClip.start, 0).toFixed(3)}s
                                <span className='SelectionChipFrames'>
                                    / {TimeUtil.calculateFrame(
                                        Math.max(displayClip.end - displayClip.start, 0),
                                        frameRate
                                    ).toLocaleString()} fr
                                </span>
                            </span>
                            <Tooltip text='Delete clip — shortcut: Delete or Backspace' placement='top'>
                                <button
                                    type='button'
                                    className='SelectionChipDelete'
                                    aria-label='Delete clip'
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        LabelActions.deleteImageLabelById(imageData.id, displayClip.id);
                                        if (activeLabelId === displayClip.id) {
                                            updateActiveLabelIdAction(null);
                                        }
                                    }}
                                >
                                    ×
                                </button>
                            </Tooltip>
                        </div>
                    )}
                    <div className='TimelineSummary'>
                        {summaryItems.map((item) => (
                            <div className='SummaryPill' key={item.label}>
                                <span>{item.label}</span>
                                <strong>{item.value}</strong>
                            </div>
                        ))}
                    </div>
                </div>
                <div className='TimelineWrapper'>
                    <CustomTimeline
                        tracks={tracks}
                        currentTime={currentTime}
                        duration={duration}
                        scaleWidth={scaleWidth}
                        onJumpToTime={onJumpToTime}
                        isCompactLayout={isCompactLayout}
                        activeLabelId={activeLabelId}
                        onSelectClip={onSelectClip}
                        playheadClock={playheadClock}
                        rowLabelsRef={rowLabelsRef}
                    />
                </div>
            </div>
            <div className='TimeInfo'>
                <div className='PlaybackState'>
                    <span className={classNames('StateDot', {active: isPlaying})} />
                    {isPlaying ? 'Playing' : 'Paused'}
                </div>
                <div className='TimeReadout'>
                    <div className='ReadoutRow'>
                        <span className='ReadoutLabel'>Time</span>
                        <span className='ReadoutValue'>{TimeUtil.formatTime(currentTime)}</span>
                        <span className='ReadoutTotal'>
                            <span className='ReadoutSep'>/</span>
                            {TimeUtil.formatTime(duration)}
                        </span>
                    </div>
                    <div className='ReadoutRow'>
                        <span className='ReadoutLabel'>Frame</span>
                        <span className='ReadoutValue'>
                            {TimeUtil.calculateFrame(currentTime, frameRate).toLocaleString()}
                        </span>
                        <span className='ReadoutTotal'>
                            <span className='ReadoutSep'>/</span>
                            {TimeUtil.calculateFrame(duration, frameRate).toLocaleString()}
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
};

const mapStateToProps = (state: AppState) => ({
    activeContext: state.general.activeContext,
    imageData: state.labels.imagesData[state.labels.activeImageIndex],
    labelNames: state.labels.labels,
    subjects: state.labels.subjects,
    activeSubjectId: state.labels.activeSubjectId,
    zoom: state.general.zoom,
    activeLabelId: state.labels.activeLabelId,
    focusedLabelNameId: state.labels.focusedLabelNameId
});

const mapDispatchToProps = {
    updateActiveLabelIdAction: updateActiveLabelId,
    updateActiveSubjectIdAction: updateActiveSubjectId
};

export default connect(
    mapStateToProps,
    mapDispatchToProps
)(EditorBottomNavigationBar);
