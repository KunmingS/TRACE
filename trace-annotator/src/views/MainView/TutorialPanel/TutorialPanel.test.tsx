import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import TutorialPanel from './TutorialPanel';

describe('TutorialPanel', () => {
    test('renders the Getting Started section by default', () => {
        render(<TutorialPanel />);

        // Header
        expect(screen.getByText('Learn TRACE')).toBeInTheDocument();
        expect(screen.getByText('Philosophy, setup, annotation, models, and CLI.')).toBeInTheDocument();

        // Section eyebrow + tabs
        expect(screen.getByText('Operator Guide')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /annotation guide/i })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /run model guide/i })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /cli guide/i })).toBeInTheDocument();

        // Getting-started block titles
        expect(screen.getByText('From video pairs to predicted annotations')).toBeInTheDocument();
        expect(screen.getByText('Pair videos with annotations')).toBeInTheDocument();
        expect(screen.getByText('One server, browser access')).toBeInTheDocument();
        expect(screen.getByText('Two ways to open recordings')).toBeInTheDocument();
        expect(screen.getByText('Four stages in sequence')).toBeInTheDocument();

        // Path specs are shown for getting-started
        expect(screen.getByText('Dataset folder')).toBeInTheDocument();
        expect(screen.getByText('Model folder')).toBeInTheDocument();
        expect(screen.getByText('Output artifacts')).toBeInTheDocument();

        // Quick-setup guide step contains a `trace app` snippet in a <code>
        const codeBlocks = document.querySelectorAll('code');
        const hasTraceApp = Array.from(codeBlocks).some((el) =>
            el.textContent?.includes('trace app')
        );
        expect(hasTraceApp).toBe(true);
    });

    test('switches between tutorial sections', () => {
        render(<TutorialPanel />);

        fireEvent.click(screen.getByRole('button', { name: /run model guide/i }));

        // Run Model section eyebrow + heading
        expect(screen.getByText('ML Workflow')).toBeInTheDocument();
        expect(
            screen.getByRole('heading', { name: 'Run Model Guide', level: 3 })
        ).toBeInTheDocument();

        // Run Model-only block + guide title
        expect(screen.getByText('Prep → Train → Test → Predict')).toBeInTheDocument();
        expect(screen.getByText('Model run')).toBeInTheDocument();

        // Getting-started content is gone
        expect(
            screen.queryByText('One server, browser access')
        ).not.toBeInTheDocument();
    });
});
