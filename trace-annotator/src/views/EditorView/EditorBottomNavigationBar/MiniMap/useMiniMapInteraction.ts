import {useCallback, useRef, useState} from 'react';

type DragMode = 'none' | 'pan' | 'resize-left' | 'resize-right';

interface UseMiniMapInteractionOptions {
    duration: number;
    scrollLeft: number;
    containerWidth: number;
    scaleWidth: number;
    startLeft: number;
    miniMapWidth: number;
    onScrollChange: (scrollLeft: number) => void;
    onZoomChange?: (zoom: number) => void;
}

interface UseMiniMapInteractionResult {
    handleMouseDown: (e: React.MouseEvent<HTMLDivElement>) => void;
    handleMouseMove: (e: React.MouseEvent<HTMLDivElement>) => void;
    handleMouseUp: () => void;
    handleMouseLeave: () => void;
    dragMode: DragMode;
}

export function useMiniMapInteraction({
    duration,
    scrollLeft,
    containerWidth,
    scaleWidth,
    startLeft,
    miniMapWidth,
    onScrollChange,
    onZoomChange,
}: UseMiniMapInteractionOptions): UseMiniMapInteractionResult {
    const [dragMode, setDragMode] = useState<DragMode>('none');
    const dragStartRef = useRef<{x: number; scrollLeft: number; scaleWidth: number}>({
        x: 0,
        scrollLeft: 0,
        scaleWidth: 0,
    });

    const totalContentWidth = startLeft + duration * scaleWidth;
    const scale = miniMapWidth / Math.max(totalContentWidth, 1);

    // Viewport indicator geometry in minimap space
    const viewportLeft = scrollLeft * scale;
    const viewportWidth = Math.max(containerWidth * scale, 6);

    const EDGE_WIDTH = 6; // px, hit target for resize handles

    const pixelToScroll = useCallback(
        (miniMapX: number): number => {
            const contentX = miniMapX / scale;
            return Math.max(0, Math.min(contentX - containerWidth / 2, totalContentWidth - containerWidth));
        },
        [scale, containerWidth, totalContentWidth]
    );

    const handleMouseDown = useCallback(
        (e: React.MouseEvent<HTMLDivElement>) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const x = e.clientX - rect.left;

            const vpLeft = viewportLeft;
            const vpRight = viewportLeft + viewportWidth;

            if (x >= vpLeft && x <= vpLeft + EDGE_WIDTH && onZoomChange) {
                // Left edge drag
                setDragMode('resize-left');
                dragStartRef.current = {x: e.clientX, scrollLeft, scaleWidth};
            } else if (x >= vpRight - EDGE_WIDTH && x <= vpRight && onZoomChange) {
                // Right edge drag
                setDragMode('resize-right');
                dragStartRef.current = {x: e.clientX, scrollLeft, scaleWidth};
            } else if (x >= vpLeft && x <= vpRight) {
                // Center drag = pan
                setDragMode('pan');
                dragStartRef.current = {x: e.clientX, scrollLeft, scaleWidth};
            } else {
                // Click outside viewport = jump
                const newScroll = pixelToScroll(x);
                onScrollChange(newScroll);
            }

            e.preventDefault();
        },
        [viewportLeft, viewportWidth, scrollLeft, scaleWidth, pixelToScroll, onScrollChange, onZoomChange]
    );

    const handleMouseMove = useCallback(
        (e: React.MouseEvent<HTMLDivElement>) => {
            if (dragMode === 'none') return;

            const dx = e.clientX - dragStartRef.current.x;

            if (dragMode === 'pan') {
                const scrollDelta = dx / scale;
                const newScroll = Math.max(
                    0,
                    Math.min(
                        dragStartRef.current.scrollLeft + scrollDelta,
                        totalContentWidth - containerWidth
                    )
                );
                onScrollChange(newScroll);
            } else if (dragMode === 'resize-left' || dragMode === 'resize-right') {
                if (!onZoomChange) return;

                // Edge drag: compute how much the viewport width changes in minimap px
                // then derive zoom factor. Moving left edge left = wider viewport = zoom out.
                // Moving right edge right = wider viewport = zoom out.
                const direction = dragMode === 'resize-left' ? -1 : 1;
                const miniDx = dx * direction;
                const newViewportWidth = Math.max(12, viewportWidth + miniDx);
                const ratio = viewportWidth / newViewportWidth;

                // scaleWidth is proportional to zoom, so scale it by ratio
                const newScaleWidth = dragStartRef.current.scaleWidth * ratio;
                // Convert scaleWidth back to approximate zoom value
                // From EditorBottomNavigationBar: scaleWidth = base + (zoom - 1) * 24
                // where base is 80 or 90. Using 85 as average.
                const baseScale = 85;
                const newZoom = Math.max(1, Math.min(4, 1 + (newScaleWidth - baseScale) / 24));
                onZoomChange(newZoom);
            }
        },
        [dragMode, scale, totalContentWidth, containerWidth, onScrollChange, onZoomChange, viewportWidth]
    );

    const handleMouseUp = useCallback(() => {
        setDragMode('none');
    }, []);

    const handleMouseLeave = useCallback(() => {
        if (dragMode !== 'none') {
            setDragMode('none');
        }
    }, [dragMode]);

    return {
        handleMouseDown,
        handleMouseMove,
        handleMouseUp,
        handleMouseLeave,
        dragMode,
    };
}
