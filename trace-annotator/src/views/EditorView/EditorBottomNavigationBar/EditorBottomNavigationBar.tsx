import React, {useMemo, useCallback} from 'react';
import './EditorBottomNavigationBar.scss';
import {ImageData, LabelName} from '../../../store/labels/types';
import {AppState} from '../../../store';
import {connect} from 'react-redux';
import {ISize} from '../../../interfaces/ISize';
import {ContextType} from '../../../data/enums/ContextType';
import classNames from 'classnames';
import {TimeUtil} from '../../../utils/TimeUtil';
import CustomTimeline from './CustomTimeline/CustomTimeline';
import {TimelineTrack} from './CustomTimeline/types';

interface IProps {
    size: ISize;
    activeContext: ContextType;
    currentTime: number;
    duration: number;
    isPlaying: boolean;
    frameRate: number;
    onTogglePlay: () => void;
    onStepFrame: (direction: 'forward' | 'backward') => void;
    onStepSeconds: (direction: 'forward' | 'backward') => void;
    onJumpToTime: (time: string) => void;
    imageData: ImageData;
    labelNames: LabelName[];
    zoom: number;
}

const EditorBottomNavigationBar: React.FC<IProps> = ({
    size,
    activeContext,
    currentTime,
    duration,
    isPlaying,
    frameRate,
    onTogglePlay,
    onStepFrame,
    onStepSeconds,
    onJumpToTime,
    imageData,
    labelNames,
    zoom
}) => {
    const formattedZoom = `${zoom.toFixed(1)}x`;
    const isCompactLayout = (size?.width || 0) < 1080 || zoom >= 2.4;
    const labelColumnWidth = isCompactLayout ? 112 : 132;
    const timeInfoWidth = isCompactLayout ? 124 : 140;
    const scaleWidth = Math.max(
        80,
        Math.min(
            160,
            (isCompactLayout ? 80 : 90) + Math.max(0, zoom - 1) * 24
        )
    );

    const tracks: TimelineTrack[] = useMemo(() => {
        const rects = imageData.labelRects || [];
        return labelNames.map(ln => {
            const clips = rects
                .filter(rect => rect.labelId === ln.id && (rect.timestamp || typeof rect.frame === 'number'))
                .map(rect => {
                    let start: number;
                    let end: number;

                    if (rect.timestamp) {
                        start = TimeUtil.parseTimestamp(rect.timestamp);
                    } else if (typeof rect.frame === 'number') {
                        start = rect.frame / frameRate;
                    } else {
                        return null;
                    }

                    const hasEndTime = rect.endTimestamp || typeof rect.endFrame === 'number';

                    if (rect.endTimestamp) {
                        end = TimeUtil.parseTimestamp(rect.endTimestamp);
                    } else if (typeof rect.endFrame === 'number') {
                        end = rect.endFrame / frameRate;
                    } else {
                        end = Math.max(currentTime, start + 0.1);
                    }

                    return {
                        id: rect.id,
                        start,
                        end,
                        labelId: ln.id,
                        color: ln.color || '#5297FF',
                        labelName: ln.name,
                        isOngoing: !hasEndTime
                    };
                })
                .filter(Boolean) as TimelineTrack['clips'];

            return {
                id: ln.id,
                name: ln.name,
                color: ln.color || '#5297FF',
                clips
            };
        });
    }, [imageData.labelRects, labelNames, frameRate, currentTime]);

    const clipCount = useMemo(() => {
        return tracks.reduce((sum, t) => sum + t.clips.length, 0);
    }, [tracks]);

    const openClipCount = (imageData.labelRects || []).filter(
        (rect) => !rect.endTimestamp && typeof rect.endFrame !== 'number'
    ).length;

    const onStepSecondsBack = useCallback(() => onStepSeconds('backward'), [onStepSeconds]);
    const onStepSecondsForward = useCallback(() => onStepSeconds('forward'), [onStepSeconds]);
    const onStepFrameBack = useCallback(() => onStepFrame('backward'), [onStepFrame]);
    const onStepFrameForward = useCallback(() => onStepFrame('forward'), [onStepFrame]);

    const summaryItems = useMemo(() => {
        const items = [
            {label: 'Tracks', value: String(labelNames.length || 0)},
            {label: 'Clips', value: String(clipCount)},
            {label: 'Zoom', value: formattedZoom}
        ];

        if (openClipCount > 0) {
            items.push({label: 'Open', value: String(openClipCount)});
        }

        return items;
    }, [clipCount, formattedZoom, labelNames.length, openClipCount]);

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
                    <span className='TimelineEyebrow'>Annotation</span>
                    <span className='TimelineSubtitle'>Tap clips to jump</span>
                </div>
                <div className='RowLabels'>
                    {labelNames.length === 0 && (
                        <div className='RowLabelsEmpty'>No behaviors</div>
                    )}
                    {labelNames.map(ln => (
                        <div key={ln.id} className='RowLabel'>
                            <div className='RowLabelDot' style={{backgroundColor: ln.color || '#5297FF'}} />
                            <span>{ln.name}</span>
                        </div>
                    ))}
                </div>
            </div>
            <div className='TimelineMain'>
                <div className='TimelineHeader'>
                    <div className='TransportControls'>
                        <button type='button' className='TransportButton' onClick={onStepSecondsBack} aria-label='Back three seconds'>
                            -3s
                        </button>
                        <button type='button' className='TransportButton' onClick={onStepFrameBack} aria-label='Previous frame'>
                            -1f
                        </button>
                        <button
                            type='button'
                            className={classNames('TransportButton', 'TransportButtonPrimary', {active: isPlaying})}
                            onClick={onTogglePlay}
                            aria-label={isPlaying ? 'Pause video' : 'Play video'}
                        >
                            {isPlaying ? 'Pause' : 'Play'}
                        </button>
                        <button type='button' className='TransportButton' onClick={onStepFrameForward} aria-label='Next frame'>
                            +1f
                        </button>
                        <button type='button' className='TransportButton' onClick={onStepSecondsForward} aria-label='Forward three seconds'>
                            +3s
                        </button>
                    </div>
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
                    />
                </div>
            </div>
            <div className='TimeInfo'>
                <div className='PlaybackState'>
                    <span className={classNames('StateDot', {active: isPlaying})} />
                    {isPlaying ? 'Playing' : 'Paused'}
                </div>
                <div className='InfoCard'>
                    <span className='InfoLabel'>Time</span>
                    <strong>{TimeUtil.formatTime(currentTime)}</strong>
                    <span className='InfoValueSubtle'>{TimeUtil.formatTime(duration)} total</span>
                </div>
                <div className='InfoCard'>
                    <span className='InfoLabel'>Frame</span>
                    <strong>{TimeUtil.calculateFrame(currentTime, frameRate)}</strong>
                    <span className='InfoValueSubtle'>{TimeUtil.calculateFrame(duration, frameRate)} total</span>
                </div>
            </div>
        </div>
    );
};

const mapStateToProps = (state: AppState) => ({
    activeContext: state.general.activeContext,
    imageData: state.labels.imagesData[state.labels.activeImageIndex],
    labelNames: state.labels.labels,
    zoom: state.general.zoom
});

export default connect(
    mapStateToProps
)(EditorBottomNavigationBar);
