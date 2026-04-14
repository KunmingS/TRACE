import React, {useMemo} from 'react';
import './TimeRuler.scss';
import {useTickGenerator} from './useTickGenerator';

interface TimeRulerProps {
    duration: number;
    scaleWidth: number;
    scrollLeft: number;
    startLeft: number;
}

const BUFFER = 100;

const TimeRuler: React.FC<TimeRulerProps> = ({
    duration,
    scaleWidth,
    scrollLeft,
    startLeft
}) => {
    const allTicks = useTickGenerator(duration, scaleWidth, startLeft);

    const ticks = useMemo(() => {
        const viewLeft = scrollLeft - BUFFER;
        const viewRight = scrollLeft + window.innerWidth + BUFFER;
        return allTicks.filter(t => t.x >= viewLeft && t.x <= viewRight);
    }, [allTicks, scrollLeft]);

    return (
        <div className="TimeRuler">
            {ticks.map((tick) => (
                <div
                    key={tick.time}
                    className={tick.isMajor ? 'TimeRuler__tick TimeRuler__tick--major' : 'TimeRuler__tick TimeRuler__tick--minor'}
                    style={{left: tick.x}}
                >
                    <div className="TimeRuler__tickLine" />
                    {tick.label !== null && (
                        <span className="TimeRuler__label">{tick.label}</span>
                    )}
                </div>
            ))}
        </div>
    );
};

export default React.memo(TimeRuler);
