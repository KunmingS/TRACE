import React from 'react';

interface Detection {
    segment: [number, number];
    label: string;
    score: number;
}

interface PipelineResultsProps {
    metrics: Record<string, any> | null;
    predictions: Record<string, Detection[]> | null;
}

const PipelineResults: React.FC<PipelineResultsProps> = ({ metrics, predictions }) => {
    if (!metrics && !predictions) return null;

    return (
        <div className='PipelineResults'>
            {metrics && (
                <div className='ResultSection'>
                    <div className='ResultSectionTitle'>Evaluation Results</div>
                    {metrics.average_mAP != null && (
                        <div className='MetricRow primary'>
                            <span>Average mAP</span>
                            <span className='MetricValue'>
                                {(metrics.average_mAP * 100).toFixed(2)}%
                            </span>
                        </div>
                    )}
                    {metrics.mAP != null && !metrics.average_mAP && (
                        <div className='MetricRow primary'>
                            <span>mAP</span>
                            <span className='MetricValue'>
                                {(metrics.mAP * 100).toFixed(2)}%
                            </span>
                        </div>
                    )}
                </div>
            )}

            {predictions && (
                <div className='ResultSection'>
                    <div className='ResultSectionTitle'>
                        Predictions
                        <span className='ResultCount'>
                            {Object.values(predictions).reduce((s, d) => s + d.length, 0)} detections
                        </span>
                    </div>
                    {Object.entries(predictions).map(([video, dets]) => (
                        <div key={video} className='VideoResult'>
                            <div className='VideoName'>{video}</div>
                            <div className='DetList'>
                                {dets.slice(0, 8).map((det, i) => (
                                    <div key={i} className='DetRow'>
                                        <span className='DetTime'>
                                            {det.segment[0].toFixed(1)}s &ndash; {det.segment[1].toFixed(1)}s
                                        </span>
                                        <span className='DetLabel'>{det.label}</span>
                                        <span className='DetScore'>
                                            {(det.score * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                ))}
                                {dets.length > 8 && (
                                    <div className='DetOverflow'>+{dets.length - 8} more</div>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default PipelineResults;
