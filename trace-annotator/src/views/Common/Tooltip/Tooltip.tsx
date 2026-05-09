import React, { useEffect, useLayoutEffect, useRef, useState } from 'react';
import ReactDOM from 'react-dom';
import './Tooltip.scss';

// A small floating tooltip that replaces the native browser `title` chrome.
// Built around two requirements that `title` couldn't meet:
//   1. Match the app's visual language (typography, color, radius).
//   2. Respect element overflow — the tooltip lives in a portal so it's
//      never clipped by an ancestor's `overflow:hidden` (which is what
//      hides the browser tooltip when chips sit in a horizontally
//      scrollable rail).
//
// Usage:
//   <Tooltip text="Hover me">
//       <button>thing</button>
//   </Tooltip>
//
// The first DOM child receives mouse listeners. To attach to multiple
// children, wrap them in a single element.

type TooltipPlacement = 'top' | 'bottom';

interface TooltipProps {
    text: React.ReactNode;
    children: React.ReactElement;
    /** Milliseconds before the tooltip appears on hover. Defaults to 350ms
     *  — long enough to avoid drive-by flicker, short enough to feel
     *  responsive when the user genuinely pauses on something. */
    delay?: number;
    /** Preferred side; the component flips when there isn't room. */
    placement?: TooltipPlacement;
    /** Maximum width in pixels. Long strings wrap to multiple lines. */
    maxWidth?: number;
    /** When false, the tooltip is suppressed entirely. Useful for
     *  conditionally-disabled tooltips (e.g., during drag operations). */
    enabled?: boolean;
}

interface Position {
    top: number;
    left: number;
    placement: TooltipPlacement;
}

const ARROW_SIZE = 5;
const VIEWPORT_PADDING = 6;

export const Tooltip: React.FC<TooltipProps> = ({
    text,
    children,
    delay = 350,
    placement = 'top',
    maxWidth = 280,
    enabled = true,
}) => {
    const [open, setOpen] = useState(false);
    const [position, setPosition] = useState<Position | null>(null);
    const tooltipRef = useRef<HTMLDivElement>(null);
    const anchorRef = useRef<HTMLElement | null>(null);
    const timerRef = useRef<number | null>(null);

    const cancelTimer = () => {
        if (timerRef.current != null) {
            window.clearTimeout(timerRef.current);
            timerRef.current = null;
        }
    };

    const scheduleOpen = () => {
        if (!enabled) return;
        cancelTimer();
        timerRef.current = window.setTimeout(() => {
            setOpen(true);
            timerRef.current = null;
        }, delay);
    };

    const closeNow = () => {
        cancelTimer();
        setOpen(false);
        setPosition(null);
    };

    // Measure & position once the tooltip mounts. We re-measure on scroll
    // and resize so the popup tracks its anchor under page changes.
    useLayoutEffect(() => {
        if (!open) return undefined;
        const recompute = () => {
            const anchor = anchorRef.current;
            const tip = tooltipRef.current;
            if (!anchor || !tip) return;
            const a = anchor.getBoundingClientRect();
            const tw = tip.offsetWidth;
            const th = tip.offsetHeight;

            // Default placement: above the anchor with a small gap.
            let chosen: TooltipPlacement = placement;
            const wantsTopY = a.top - th - ARROW_SIZE - 2;
            const wantsBottomY = a.bottom + ARROW_SIZE + 2;
            if (chosen === 'top' && wantsTopY < VIEWPORT_PADDING) chosen = 'bottom';
            if (chosen === 'bottom' && wantsBottomY + th > window.innerHeight - VIEWPORT_PADDING) {
                // No room below either — pick whichever has more space.
                chosen = a.top > window.innerHeight - a.bottom ? 'top' : 'bottom';
            }
            const top = chosen === 'top' ? wantsTopY : wantsBottomY;

            // Center horizontally, then clamp to viewport.
            let left = a.left + a.width / 2 - tw / 2;
            const maxLeft = window.innerWidth - tw - VIEWPORT_PADDING;
            if (left < VIEWPORT_PADDING) left = VIEWPORT_PADDING;
            else if (left > maxLeft) left = maxLeft;

            setPosition({ top, left, placement: chosen });
        };
        recompute();
        window.addEventListener('scroll', recompute, true);
        window.addEventListener('resize', recompute);
        return () => {
            window.removeEventListener('scroll', recompute, true);
            window.removeEventListener('resize', recompute);
        };
    }, [open, placement, text]);

    // Hide on Escape so users can dismiss without moving the mouse.
    useEffect(() => {
        if (!open) return undefined;
        const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') closeNow(); };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [open]);

    useEffect(() => () => cancelTimer(), []);

    // Inject hover listeners on whatever child the caller passes in.
    // Cloning preserves their existing handlers — we just compose ours on top.
    const child = React.Children.only(children);
    const childWithHandlers = React.cloneElement(child as React.ReactElement<any>, {
        ref: (node: HTMLElement | null) => {
            anchorRef.current = node;
            // Forward refs that already exist on the child so callers don't
            // lose theirs by wrapping with Tooltip.
            const childRef = (child as any).ref;
            if (typeof childRef === 'function') childRef(node);
            else if (childRef && typeof childRef === 'object') childRef.current = node;
        },
        onMouseEnter: (e: React.MouseEvent) => {
            scheduleOpen();
            (child.props as any).onMouseEnter?.(e);
        },
        onMouseLeave: (e: React.MouseEvent) => {
            closeNow();
            (child.props as any).onMouseLeave?.(e);
        },
        onFocus: (e: React.FocusEvent) => {
            // Show immediately on keyboard focus — there's no "drive-by"
            // hover problem when the user has tabbed onto the element.
            cancelTimer();
            setOpen(true);
            (child.props as any).onFocus?.(e);
        },
        onBlur: (e: React.FocusEvent) => {
            closeNow();
            (child.props as any).onBlur?.(e);
        },
        onMouseDown: (e: React.MouseEvent) => {
            // Tooltip should disappear when the user commits to clicking;
            // it would otherwise float over whatever modal/menu opens.
            closeNow();
            (child.props as any).onMouseDown?.(e);
        },
    });

    const portal = open && text != null && text !== ''
        ? ReactDOM.createPortal(
            <div
                ref={tooltipRef}
                className={`Tooltip${position ? ` placement-${position.placement}` : ' measuring'}`}
                role='tooltip'
                style={{
                    top: position?.top ?? -9999,
                    left: position?.left ?? -9999,
                    maxWidth,
                }}
            >
                {text}
            </div>,
            document.body,
        )
        : null;

    return (
        <>
            {childWithHandlers}
            {portal}
        </>
    );
};

export default Tooltip;
