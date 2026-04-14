import React from 'react';
import { PipelineStepId, StepState, StepStatus } from './usePipelineState';

interface PipelineProgressProps {
    stepStates: Record<PipelineStepId, StepState>;
    currentStep: PipelineStepId | null;
    enabledSteps: { train: boolean; test: boolean; infer: boolean };
}

const STEP_LABELS: Record<PipelineStepId, string> = {
    prep: 'Prepare Dataset',
    train: 'Train Model',
    test: 'Evaluate',
    infer: 'Run Inference',
};

const statusIcon = (status: StepStatus): string => {
    switch (status) {
        case 'completed': return '\u2713';
        case 'failed': return '\u2717';
        case 'cancelled': return '\u2013';
        case 'running': return '\u25CF';
        case 'skipped': return '\u2013';
        default: return '\u25CB';
    }
};

const PipelineProgress: React.FC<PipelineProgressProps> = ({
    stepStates,
    currentStep,
    enabledSteps,
}) => {
    const visibleSteps: PipelineStepId[] = [];
    // Always show prep if any dataset step is enabled
    if (enabledSteps.train || enabledSteps.test) visibleSteps.push('prep');
    if (enabledSteps.train) visibleSteps.push('train');
    if (enabledSteps.test) visibleSteps.push('test');
    if (enabledSteps.infer) visibleSteps.push('infer');

    if (visibleSteps.length === 0) return null;

    return (
        <div className='PipelineProgress'>
            {visibleSteps.map((stepId, i) => {
                const state = stepStates[stepId];
                const isActive = stepId === currentStep;
                const isLast = i === visibleSteps.length - 1;

                return (
                    <div
                        key={stepId}
                        className={`ProgressStep ${state.status} ${isActive ? 'active' : ''}`}
                    >
                        <div className='StepSpine'>
                            <span className={`StepDot ${state.status}`}>
                                {statusIcon(state.status)}
                            </span>
                            {!isLast && <span className='StepLine' />}
                        </div>
                        <div className='StepContent'>
                            <span className='StepLabel'>{STEP_LABELS[stepId]}</span>
                            {state.error && (
                                <span className='StepError'>{state.error}</span>
                            )}
                        </div>
                    </div>
                );
            })}
        </div>
    );
};

export default PipelineProgress;
