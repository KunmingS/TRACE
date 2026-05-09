import React, { useRef, useEffect, useMemo } from 'react';
import { ISize } from '../../../../interfaces/ISize';
import { ImageData, LabelName, LabelRect } from '../../../../store/labels/types';
import './RectLabelsList.scss';
import {
    updateActiveLabelId,
    updateActiveLabelNameId,
    updateImageDataById
} from '../../../../store/labels/actionCreators';
import { updateLabelNames } from '../../../../store/labels/actionCreators';
import { AppState } from '../../../../store';
import { connect } from 'react-redux';
import { LabelActions } from '../../../../logic/actions/LabelActions';
import { TimeUtil } from '../../../../utils/TimeUtil';

interface IProps {
    size: ISize;
    imageData: ImageData;
    updateImageDataByIdAction: (id: string, newImageData: ImageData) => any;
    activeLabelId: string;
    highlightedLabelId: string;
    updateActiveLabelNameIdAction: (activeLabelId: string) => any;
    labelNames: LabelName[];
    updateActiveLabelIdAction: (activeLabelId: string) => any;
    updateLabelNamesAction: (labels: LabelName[]) => any;
    onJumpToTime?: (timestamp: string) => void;
}

const RectLabelsList: React.FC<IProps> = ({
    size,
    imageData,
    updateImageDataByIdAction,
    labelNames,
    updateLabelNamesAction,
    updateActiveLabelNameIdAction,
    activeLabelId,
    highlightedLabelId,
    updateActiveLabelIdAction,
    onJumpToTime
}) => {
    const rowHeight = 32;
    const listRef = useRef<HTMLDivElement>(null);

    // Pre-parse timestamps once per labelRects change so the per-tick scroll
    // effect below doesn't re-parse every timestamp string at 10Hz.
    const sortedEntries = useMemo(() => {
        return imageData.labelRects
            .map(rect => ({
                rect,
                startTime: TimeUtil.parseTimestamp(rect.timestamp!),
                endTime: rect.endTimestamp
                    ? TimeUtil.parseTimestamp(rect.endTimestamp)
                    : Infinity,
            }))
            .sort((a, b) => a.startTime - b.startTime);
    }, [imageData.labelRects]);

    const sortedLabelRects = useMemo(
        () => sortedEntries.map(e => e.rect),
        [sortedEntries]
    );

    // O(1) label-name lookup — replaces a per-row findLast scan (was O(n²) overall).
    const labelNameById = useMemo(() => {
        const map = new Map<string, LabelName>();
        for (const l of labelNames) map.set(l.id, l);
        return map;
    }, [labelNames]);

    useEffect(() => {
        if (!listRef.current) return;
        if (sortedEntries.length === 0) return;
        const currentTime = imageData.timestamp;
        // Prefer a clip that contains t; else fall back to the most recent past clip.
        let idx = sortedEntries.findIndex(
            e => e.startTime <= currentTime && currentTime <= e.endTime
        );
        if (idx < 0) {
            // Scan backward for the last entry with startTime <= currentTime.
            for (let i = sortedEntries.length - 1; i >= 0; i--) {
                if (sortedEntries[i].startTime <= currentTime) { idx = i; break; }
            }
        }
        if (idx >= 0) {
            listRef.current.scrollTop = idx * rowHeight;
        }
    }, [sortedEntries, imageData.timestamp, rowHeight]);

    const deleteRect = (id: string) => {
        LabelActions.deleteRectLabelById(imageData.id, id);
    };

    const formatTs = (ts: string) => {
        try {
            const seconds = TimeUtil.parseTimestamp(ts);
            return TimeUtil.formatTime(seconds);
        } catch { return ts; }
    };

    const getDuration = (start: string, end: string) => {
        try {
            const s = TimeUtil.parseTimestamp(start);
            const e = TimeUtil.parseTimestamp(end);
            return (e - s).toFixed(1) + 's';
        } catch { return ''; }
    };

    const handleRowClick = (rect: LabelRect) => {
        updateActiveLabelIdAction(rect.id);
        if (onJumpToTime && rect.timestamp) {
            onJumpToTime(rect.timestamp);
        }
    };

    if (imageData.labelRects.length === 0) {
        return (
            <div className="ClipsTable" style={{ width: size.width }}>
                <div className="EmptyState">No behavior clips annotated yet</div>
            </div>
        );
    }

    return (
        <div className="ClipsTable" ref={listRef} style={{ width: size.width, maxHeight: size.height }}>
            {sortedLabelRects.map(rect => {
                const labelName = rect.labelId ? labelNameById.get(rect.labelId) : null;
                const isActive = rect.id === activeLabelId;
                const isOngoing = !rect.endTimestamp;

                return (
                    <div
                        key={rect.id}
                        className={`ClipRow ${isActive ? 'active' : ''} ${isOngoing ? 'ongoing' : ''}`}
                        onClick={() => handleRowClick(rect)}
                    >
                        <div
                            className="ColorDot"
                            style={{ backgroundColor: labelName?.color || '#5297FF' }}
                        />
                        <span className="BehaviorName">
                            {labelName?.name || rect.behavior || 'Behavior'}
                        </span>
                        <span
                            className="Timestamp"
                            onClick={e => {
                                e.stopPropagation();
                                if (onJumpToTime && rect.timestamp) onJumpToTime(rect.timestamp);
                            }}
                        >
                            {formatTs(rect.timestamp!)}
                        </span>
                        <span className="Arrow">&rarr;</span>
                        {rect.endTimestamp ? (
                            <>
                                <span
                                    className="Timestamp"
                                    onClick={e => {
                                        e.stopPropagation();
                                        if (onJumpToTime) onJumpToTime(rect.endTimestamp!);
                                    }}
                                >
                                    {formatTs(rect.endTimestamp)}
                                </span>
                                <span className="Duration">
                                    {getDuration(rect.timestamp!, rect.endTimestamp)}
                                </span>
                            </>
                        ) : (
                            <span className="OngoingLabel">recording...</span>
                        )}
                        <div className="Actions">
                            <button
                                type='button'
                                className="DeleteBtn"
                                onClick={e => { e.stopPropagation(); deleteRect(rect.id); }}
                                title="Delete clip"
                                aria-label="Delete clip"
                            >
                                &times;
                            </button>
                        </div>
                    </div>
                );
            })}
        </div>
    );
};

const mapDispatchToProps = {
    updateImageDataByIdAction: updateImageDataById,
    updateActiveLabelNameIdAction: updateActiveLabelNameId,
    updateActiveLabelIdAction: updateActiveLabelId,
    updateLabelNamesAction: updateLabelNames,
};

const mapStateToProps = (state: AppState) => ({
    activeLabelId: state.labels.activeLabelId,
    highlightedLabelId: state.labels.highlightedLabelId,
    labelNames: state.labels.labels,
    imageData: state.labels.imagesData[state.labels.activeImageIndex]
});

export default connect(
    mapStateToProps,
    mapDispatchToProps
)(RectLabelsList);
