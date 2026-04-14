import React, {useMemo, useRef, useCallback} from 'react';
import './MiniMap.scss';
import {TimelineTrack} from '../CustomTimeline/types';
import {useMiniMapInteraction} from './useMiniMapInteraction';
import classNames from 'classnames';
import {store} from '../../../..';
import {updateZoom} from '../../../../store/general/actionCreators';
import {ViewPointSettings} from '../../../../settings/ViewPointSettings';

interface MiniMapProps {
    duration: number;
    tracks: TimelineTrack[];
    scrollLeft: number;
    containerWidth: number;
    scaleWidth: number;
    startLeft: number;
    onScrollChange: (scrollLeft: number) => void;
}

const MiniMap: React.FC<MiniMapProps> = ({
    duration,
    tracks,
    scrollLeft,
    containerWidth,
    scaleWidth,
    startLeft,
    onScrollChange,
}) => {
    const containerRef = useRef<HTMLDivElement>(null);

    const miniMapWidth = containerRef.current?.clientWidth ?? 400;
    const totalContentWidth = startLeft + duration * scaleWidth;
    const scale = miniMapWidth / Math.max(totalContentWidth, 1);

    const viewportLeft = scrollLeft * scale;
    const viewportWidth = Math.max(containerWidth * scale, 6);

    const handleZoomChange = useCallback((newZoom: number) => {
        const clamped = Math.max(
            ViewPointSettings.MIN_ZOOM,
            Math.min(ViewPointSettings.MAX_ZOOM, newZoom)
        );
        store.dispatch(updateZoom(clamped));
    }, []);

    const {handleMouseDown, handleMouseMove, handleMouseUp, handleMouseLeave, dragMode} =
        useMiniMapInteraction({
            duration,
            scrollLeft,
            containerWidth,
            scaleWidth,
            startLeft,
            miniMapWidth,
            onScrollChange,
            onZoomChange: handleZoomChange,
        });

    const clips = useMemo(() => {
        const result: {left: number; width: number; color: string; key: string}[] = [];
        for (const track of tracks) {
            for (const clip of track.clips) {
                const clipLeft = ((startLeft + clip.start * scaleWidth) / Math.max(totalContentWidth, 1)) * miniMapWidth;
                const clipRight = ((startLeft + clip.end * scaleWidth) / Math.max(totalContentWidth, 1)) * miniMapWidth;
                result.push({
                    left: clipLeft,
                    width: Math.max(clipRight - clipLeft, 2),
                    color: clip.color,
                    key: clip.id,
                });
            }
        }
        return result;
    }, [tracks, startLeft, scaleWidth, totalContentWidth, miniMapWidth]);

    return (
        <div
            ref={containerRef}
            className={classNames('MiniMap', {
                'dragging-pan': dragMode === 'pan',
                'dragging-resize': dragMode === 'resize-left' || dragMode === 'resize-right',
            })}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseLeave}
        >
            {clips.map(clip => (
                <div
                    key={clip.key}
                    className="MiniMapClip"
                    style={{
                        left: clip.left,
                        width: clip.width,
                        backgroundColor: clip.color,
                    }}
                />
            ))}
            <div
                className={classNames('MiniMapViewport', {
                    dragging: dragMode !== 'none',
                })}
                style={{
                    left: viewportLeft,
                    width: viewportWidth,
                }}
            />
        </div>
    );
};

export default React.memo(MiniMap);
