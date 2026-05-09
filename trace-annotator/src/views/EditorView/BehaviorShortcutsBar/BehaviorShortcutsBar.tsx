import React, {useMemo, useState, useEffect, useRef} from 'react';
import { connect } from 'react-redux';
import { AppState } from '../../../store';
import { ImageData, LabelName, Subject } from '../../../store/labels/types';
import { updateActivePopupType } from '../../../store/general/actionCreators';
import { updateActiveSubjectId, updateFocusedLabelNameId } from '../../../store/labels/actionCreators';
import { PopupWindowType } from '../../../data/enums/PopupWindowType';
import { setPendingFocusLabelId } from '../../PopupView/InsertLabelNamesPopup/focusState';
import { DEFAULT_SUBJECT_ID } from '../../../store/labels/reducer';
import { TimeUtil } from '../../../utils/TimeUtil';
import { PlayheadClock } from '../EditorBottomNavigationBar/playheadClock';
import Tooltip from '../../Common/Tooltip/Tooltip';
import './BehaviorShortcutsBar.scss';

interface IProps {
    imageData: ImageData;
    labelNames: LabelName[];
    subjects: Subject[];
    activeSubjectId: string | null;
    activeLabelId: string | null;
    focusedLabelNameId: string | null;
    playheadClock?: PlayheadClock;
    updateActivePopupTypeAction: (popupType: PopupWindowType) => any;
    updateActiveSubjectIdAction: (id: string | null) => any;
    updateFocusedLabelNameIdAction: (id: string | null) => any;
}

const BehaviorShortcutsBar: React.FC<IProps> = ({
    imageData,
    labelNames,
    subjects,
    activeSubjectId,
    activeLabelId,
    focusedLabelNameId,
    playheadClock,
    updateActivePopupTypeAction,
    updateActiveSubjectIdAction,
    updateFocusedLabelNameIdAction,
}) => {
    const [pickerOpen, setPickerOpen] = useState(false);
    const pickerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!pickerOpen) return undefined;
        const onDoc = (e: MouseEvent) => {
            if (pickerRef.current && !pickerRef.current.contains(e.target as Node)) {
                setPickerOpen(false);
            }
        };
        document.addEventListener('mousedown', onDoc);
        return () => document.removeEventListener('mousedown', onDoc);
    }, [pickerOpen]);
    const openEditorFocused = (labelId: string) => {
        setPendingFocusLabelId(labelId);
        updateActivePopupTypeAction(PopupWindowType.UPDATE_LABEL);
    };

    const openEditorForNew = () => {
        setPendingFocusLabelId(null);
        updateActivePopupTypeAction(
            labelNames.length === 0
                ? PopupWindowType.INSERT_LABEL_NAMES
                : PopupWindowType.UPDATE_LABEL
        );
    };

    // O(rects) once per render instead of O(labels × rects) inside the loop.
    // Scoped to the active subject so the chip's "recording" state mirrors
    // what the next behavior-key press will toggle.
    const recordingLabelIds = useMemo(() => {
        const s = new Set<string>();
        const subjectId = activeSubjectId ?? DEFAULT_SUBJECT_ID;
        for (const rect of imageData?.labelRects || []) {
            if (rect.labelId
                && !rect.endTimestamp
                && (rect.animalId ?? DEFAULT_SUBJECT_ID) === subjectId) {
                s.add(rect.labelId);
            }
        }
        return s;
    }, [imageData?.labelRects, activeSubjectId]);

    // Which behavior is the currently-selected clip an instance of? Drives
    // the `.selected` highlight on the matching chip so the user gets a
    // visual link between the timeline selection and the behavior bar.
    const selectedBehaviorId = useMemo(() => {
        if (!activeLabelId) return null;
        const rect = (imageData?.labelRects || []).find(r => r.id === activeLabelId);
        return rect?.labelId ?? null;
    }, [activeLabelId, imageData?.labelRects]);

    // Pre-built [start, end, labelId] triples for the active subject's
    // clips, sorted by start time. Used by the playhead-watcher below to
    // figure out which behaviors a moving playhead is currently inside —
    // and by the chip click counter so the tooltip can show "N clips".
    const clipIntervals = useMemo(() => {
        const subjectId = activeSubjectId ?? DEFAULT_SUBJECT_ID;
        const out: Array<{ labelId: string; start: number; end: number }> = [];
        for (const rect of imageData?.labelRects || []) {
            if (!rect.labelId || !rect.timestamp) continue;
            if ((rect.animalId ?? DEFAULT_SUBJECT_ID) !== subjectId) continue;
            let start: number, end: number;
            try {
                start = TimeUtil.parseTimestamp(rect.timestamp);
                end = rect.endTimestamp
                    ? TimeUtil.parseTimestamp(rect.endTimestamp)
                    : start + 1 / (imageData?.frameRate ?? 30);
            } catch { continue; }
            out.push({ labelId: rect.labelId, start, end });
        }
        out.sort((a, b) => a.start - b.start);
        return out;
    }, [imageData?.labelRects, imageData?.frameRate, activeSubjectId]);

    const clipCountByBehavior = useMemo(() => {
        const counts = new Map<string, number>();
        for (const c of clipIntervals) counts.set(c.labelId, (counts.get(c.labelId) ?? 0) + 1);
        return counts;
    }, [clipIntervals]);

    // Behavior(s) whose clip currently sits under the playhead. We update
    // this every frame via rAF, but only commit a setState when the *set*
    // changes — typically only at clip boundaries — so we don't churn the
    // tree at 30+Hz during normal playback.
    const [playheadActiveIds, setPlayheadActiveIds] = useState<Set<string>>(new Set());
    useEffect(() => {
        if (!playheadClock) return undefined;
        let rafId: number;
        const tick = () => {
            const t = playheadClock.current;
            const next = new Set<string>();
            // O(N) over clips of the active subject. Fine for N≈100 at 60Hz;
            // if it ever becomes a bottleneck we can binary-search by start.
            for (const c of clipIntervals) {
                if (t >= c.start && t <= c.end) next.add(c.labelId);
            }
            setPlayheadActiveIds(prev => {
                if (prev.size === next.size) {
                    let same = true;
                    for (const id of prev) { if (!next.has(id)) { same = false; break; } }
                    if (same) return prev;
                }
                return next;
            });
            rafId = requestAnimationFrame(tick);
        };
        rafId = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(rafId);
    }, [playheadClock, clipIntervals]);

    // If the focused behavior is deleted (or labels reload without it),
    // exit focus mode automatically — otherwise the timeline would be
    // pinned to a label that no longer exists.
    useEffect(() => {
        if (focusedLabelNameId && !labelNames.some(l => l.id === focusedLabelNameId)) {
            updateFocusedLabelNameIdAction(null);
        }
    }, [focusedLabelNameId, labelNames, updateFocusedLabelNameIdAction]);

    // Plain click on a chip toggles focus mode for that behavior.
    // Click again on the focused chip → exit focus.
    const toggleFocus = (labelId: string) => {
        updateFocusedLabelNameIdAction(focusedLabelNameId === labelId ? null : labelId);
    };

    // The chip's `:focus-visible` style draws a 4px ring in the chip's own
    // color, which is visually indistinguishable from `.playhead-active` /
    // `.selected`. Browsers can keep that ring after a click whenever the
    // page has been touched by keyboard recently (e.g. behavior shortcut
    // keys, Tab focus), so the chip looks "lit" even when the playhead is
    // on blank space. Drop focus right after the toggle so only
    // `.playhead-active` / `.selected` / `.focused` decide whether the chip
    // is highlighted.
    const handleChipClick = (e: React.MouseEvent<HTMLButtonElement>, labelId: string) => {
        toggleFocus(labelId);
        e.currentTarget.blur();
    };

    const activeSubject = subjects.find(s => s.id === activeSubjectId)
        ?? subjects.find(s => s.id === DEFAULT_SUBJECT_ID)
        ?? subjects[0]
        ?? null;
    const showSubjectPill = subjects.length > 1;
    const activeIndex = activeSubject ? subjects.findIndex(s => s.id === activeSubject.id) : -1;

    return (
        <div className='BehaviorShortcutsBar'>
            {showSubjectPill && activeSubject && (
                <div className='SubjectPickerWrap' ref={pickerRef}>
                    <Tooltip text='Switch active subject (press 1–9)'>
                    <button
                        type='button'
                        className='SubjectPill'
                        onClick={() => setPickerOpen(o => !o)}
                        aria-haspopup='listbox'
                        aria-expanded={pickerOpen}
                    >
                        <span className='SubjectPillEyebrow'>Recording for</span>
                        <span className='SubjectPillDigit'>{activeIndex + 1}</span>
                        <span className='SubjectPillName'>{activeSubject.name}</span>
                        <svg className='SubjectPillCaret' width='10' height='10' viewBox='0 0 10 10' fill='none'>
                            <path d='M2 4l3 3 3-3' stroke='currentColor' strokeWidth='1.4' strokeLinecap='round' strokeLinejoin='round'/>
                        </svg>
                    </button>
                    </Tooltip>
                    {pickerOpen && (
                        <div className='SubjectPickerMenu' role='listbox'>
                            {subjects.map((s, i) => (
                                <button
                                    type='button'
                                    key={s.id}
                                    role='option'
                                    aria-selected={s.id === activeSubject.id}
                                    className={`SubjectPickerOption${s.id === activeSubject.id ? ' active' : ''}`}
                                    onClick={() => {
                                        updateActiveSubjectIdAction(s.id);
                                        setPickerOpen(false);
                                    }}
                                >
                                    <span className='SubjectPickerDigit'>{i + 1}</span>
                                    <span className='SubjectPickerName'>{s.name}</span>
                                </button>
                            ))}
                        </div>
                    )}
                </div>
            )}
            <span className='BarLabel'>{focusedLabelNameId ? 'Focus' : 'Behaviors'}</span>
            <div className={`ChipRailWrap${focusedLabelNameId ? ' focused' : ''}`}>
                <div className='ChipRail'>
                    {labelNames
                        // In focus mode, hide every chip except the focused one
                        // — the user asked for the rest to "not exist anymore".
                        // Other chips remain in the React tree only when no
                        // focus is active.
                        .filter(ln => !focusedLabelNameId || ln.id === focusedLabelNameId)
                        .map(ln => {
                        const color = ln.color || '#5297FF';
                        const recording = recordingLabelIds.has(ln.id);
                        const selected = ln.id === selectedBehaviorId;
                        const playing = playheadActiveIds.has(ln.id);
                        const focused = ln.id === focusedLabelNameId;
                        const clipCount = clipCountByBehavior.get(ln.id) ?? 0;
                        const tipText = focused
                            ? `Click to exit focus mode — Shift+←/→ to jump between “${ln.name}” clips, Cmd/Ctrl+←/→ to nudge endpoints`
                            : clipCount > 0
                                ? `Click to focus on “${ln.name}” (${clipCount} clip${clipCount === 1 ? '' : 's'}) — pencil to edit`
                                : `Click to focus on “${ln.name}” (no clips yet) — pencil to edit`;
                        return (
                            <Tooltip key={ln.id} text={tipText}>
                                <button
                                    type='button'
                                    className={
                                        `BehaviorChip` +
                                        (recording ? ' recording' : '') +
                                        (selected ? ' selected' : '') +
                                        (playing ? ' playhead-active' : '') +
                                        (focused ? ' focused' : '')
                                    }
                                    onClick={(e) => handleChipClick(e, ln.id)}
                                    style={{
                                        ['--chip-color' as string]: color,
                                        ['--chip-tint' as string]: `${color}14`,
                                        ['--chip-tint-strong' as string]: `${color}26`
                                    }}
                                >
                                    <span className='ChipDot' />
                                    {ln.shortcut ? (
                                        <span className='ChipKey'>{ln.shortcut}</span>
                                    ) : (
                                        <span className='ChipKey ChipKeyMissing' aria-label='Shortcut not set'>·</span>
                                    )}
                                    <span className='ChipName'>{ln.name}</span>
                                    {recording && <span className='ChipRecDot' aria-label='Recording' />}
                                    {focused && <span className='ChipFocusDot' aria-label='Focused — click to exit' />}
                                    <span
                                        className='ChipEditGlyph'
                                        role='button'
                                        aria-label={`Edit ${ln.name}`}
                                        onClick={(e) => { e.stopPropagation(); openEditorFocused(ln.id); }}
                                    >
                                        <svg width='11' height='11' viewBox='0 0 11 11' fill='none'>
                                            <path d='M1.5 7.9L7.6 1.8a1 1 0 0 1 1.4 0l.2.2a1 1 0 0 1 0 1.4L3.1 9.5l-2 .4.4-2z'
                                                stroke='currentColor' strokeWidth='1.1' strokeLinejoin='round'/>
                                        </svg>
                                    </span>
                                </button>
                            </Tooltip>
                        );
                    })}
                    {/* "New type" sits inside the rail so it scrolls with the
                        chips and is visually adjacent to the last behavior.
                        Hidden in focus mode so the focused chip stands alone. */}
                    {!focusedLabelNameId && (
                        <Tooltip text='Create a new behavior type'>
                            <button
                                type='button'
                                className='NewBehaviorChip'
                                onClick={openEditorForNew}
                            >
                                <span className='NewGlyph' aria-hidden='true'>
                                    <svg width='11' height='11' viewBox='0 0 11 11' fill='none'>
                                        <path d='M5.5 1.6v7.8M1.6 5.5h7.8' stroke='currentColor' strokeWidth='1.4' strokeLinecap='round'/>
                                    </svg>
                                </span>
                                <span className='NewLabel'>New type</span>
                            </button>
                        </Tooltip>
                    )}
                    {focusedLabelNameId && (
                        <span className='FocusHint' aria-live='polite'>
                            Other behaviors hidden — click chip again to exit
                        </span>
                    )}
                </div>
            </div>
        </div>
    );
};

const mapStateToProps = (state: AppState) => ({
    imageData: state.labels.imagesData[state.labels.activeImageIndex],
    labelNames: state.labels.labels,
    subjects: state.labels.subjects,
    activeSubjectId: state.labels.activeSubjectId,
    activeLabelId: state.labels.activeLabelId,
    focusedLabelNameId: state.labels.focusedLabelNameId,
});

const mapDispatchToProps = {
    updateActivePopupTypeAction: updateActivePopupType,
    updateActiveSubjectIdAction: updateActiveSubjectId,
    updateFocusedLabelNameIdAction: updateFocusedLabelNameId,
};

export default connect(mapStateToProps, mapDispatchToProps)(React.memo(BehaviorShortcutsBar));
