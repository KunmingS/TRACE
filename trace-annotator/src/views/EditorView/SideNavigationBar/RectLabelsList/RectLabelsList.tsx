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
import { findLast } from 'lodash';
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

    const sortedLabelRects = useMemo(() =>
        [...imageData.labelRects].sort((a, b) =>
            TimeUtil.parseTimestamp(a.timestamp!) - TimeUtil.parseTimestamp(b.timestamp!)
        ),
        [imageData.labelRects]
    );

    useEffect(() => {
        if (!listRef.current) return;
        const currentTime = imageData.timestamp;
        let idx = sortedLabelRects.findIndex(r => {
            const start = TimeUtil.parseTimestamp(r.timestamp!);
            const end = r.endTimestamp ? TimeUtil.parseTimestamp(r.endTimestamp) : Infinity;
            return start <= currentTime && currentTime <= end;
        });
        if (idx < 0) {
            idx = sortedLabelRects.filter(r => TimeUtil.parseTimestamp(r.timestamp!) <= currentTime).length - 1;
        }
        if (idx >= 0) {
            listRef.current.scrollTop = idx * rowHeight;
        }
    }, [sortedLabelRects, imageData.timestamp, rowHeight]);

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
                const labelName = rect.labelId ? findLast(labelNames, { id: rect.labelId }) : null;
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
