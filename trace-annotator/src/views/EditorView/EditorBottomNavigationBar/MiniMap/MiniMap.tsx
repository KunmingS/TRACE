import React, {useMemo, useRef, useCallback, useEffect, useState} from 'react';
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

// Minimap shows this many times the viewport width of content — keeps the
// viewport indicator visibly sized on long videos instead of shrinking to a
// sliver. When the video is shorter than this window, the full video is shown.
const MINIMAP_WINDOW_RATIO = 5;

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
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [miniMapSize, setMiniMapSize] = useState({width: 400, height: 28});

    const miniMapWidth = miniMapSize.width;
    const miniMapHeight = miniMapSize.height;
    const totalContentWidth = startLeft + duration * scaleWidth;

    const desiredWindowPx = Math.max(containerWidth, 1) * MINIMAP_WINDOW_RATIO;
    const miniMapRangePx = Math.max(1, Math.min(totalContentWidth, desiredWindowPx));
    const viewportCenter = scrollLeft + containerWidth / 2;
    const maxWindowLeft = Math.max(0, totalContentWidth - miniMapRangePx);
    const windowLeft = Math.max(0, Math.min(maxWindowLeft, viewportCenter - miniMapRangePx / 2));
    const scale = miniMapWidth / miniMapRangePx;
    const isWindowed = totalContentWidth > miniMapRangePx + 0.5;

    const viewportLeft = (scrollLeft - windowLeft) * scale;
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
            scrollLeft,
            containerWidth,
            scaleWidth,
            totalContentWidth,
            windowLeft,
            scale,
            onScrollChange,
            onZoomChange: handleZoomChange,
        });

    useEffect(() => {
        const container = containerRef.current;
        if (!container) return undefined;

        const updateSize = () => {
            const nextWidth = container.clientWidth || 400;
            const nextHeight = container.clientHeight || 28;
            setMiniMapSize((prev) => (
                prev.width === nextWidth && prev.height === nextHeight
                    ? prev
                    : {width: nextWidth, height: nextHeight}
            ));
        };

        const observer = new ResizeObserver(updateSize);
        observer.observe(container);
        updateSize();

        return () => observer.disconnect();
    }, []);

    const clipRects = useMemo(() => {
        const result: {left: number; width: number; color: string; key: string}[] = [];
        for (const track of tracks) {
            for (const clip of track.clips) {
                const clipLeftContent = startLeft + clip.start * scaleWidth;
                const clipRightContent = startLeft + clip.end * scaleWidth;
                const clipLeftPx = (clipLeftContent - windowLeft) * scale;
                const clipRightPx = (clipRightContent - windowLeft) * scale;
                if (clipRightPx < 0 || clipLeftPx > miniMapWidth) continue;
                const visibleLeft = Math.max(0, clipLeftPx);
                const visibleRight = Math.min(miniMapWidth, clipRightPx);
                result.push({
                    left: visibleLeft,
                    width: Math.max(visibleRight - visibleLeft, 2),
                    color: clip.color,
                    key: clip.id,
                });
            }
        }
        return result;
    }, [tracks, startLeft, scaleWidth, windowLeft, scale, miniMapWidth]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || miniMapWidth <= 0 || miniMapHeight <= 0) return;

        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.max(1, Math.round(miniMapWidth * dpr));
        canvas.height = Math.max(1, Math.round(miniMapHeight * dpr));
        canvas.style.width = `${miniMapWidth}px`;
        canvas.style.height = `${miniMapHeight}px`;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        const top = 2;
        const height = Math.max(miniMapHeight - 4, 1);
        ctx.globalAlpha = 0.6;

        for (const clip of clipRects) {
            ctx.fillStyle = clip.color;
            ctx.fillRect(clip.left, top, clip.width, height);
        }

        ctx.globalAlpha = 1;
    }, [clipRects, miniMapHeight, miniMapWidth]);

    const hasContentBeforeWindow = isWindowed && windowLeft > 0.5;
    const hasContentAfterWindow = isWindowed && windowLeft + miniMapRangePx < totalContentWidth - 0.5;

    return (
        <div
            ref={containerRef}
            className={classNames('MiniMap', {
                'dragging-pan': dragMode === 'pan',
                'dragging-resize': dragMode === 'resize-left' || dragMode === 'resize-right',
                'fade-left': hasContentBeforeWindow,
                'fade-right': hasContentAfterWindow,
            })}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseLeave}
        >
            <canvas className="MiniMapCanvas" ref={canvasRef} />
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
