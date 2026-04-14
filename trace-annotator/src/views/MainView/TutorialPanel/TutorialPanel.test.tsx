import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import TutorialPanel from './TutorialPanel';

describe('TutorialPanel', () => {
    test('renders the structured getting started guide by default', () => {
        render(<TutorialPanel />);

        expect(screen.getByText('Learn TRACE')).toBeInTheDocument();
        expect(screen.getByText('One lab server, browser access from every rig')).toBeInTheDocument();
        expect(screen.getByText('One backend, three control surfaces')).toBeInTheDocument();
        expect(screen.getByText('Web page')).toBeInTheDocument();
        expect(screen.getByText('Server')).toBeInTheDocument();
        expect(screen.getByText('Command line')).toBeInTheDocument();
        expect(screen.getByText('Choose how a recording session enters TRACE')).toBeInTheDocument();
        expect(screen.getByText('Build your own task recipe')).toBeInTheDocument();
        expect(screen.getByText('System, pipeline, and CLI')).toBeInTheDocument();
        expect(screen.getByText('Dataset, model, and input folders')).toBeInTheDocument();
        expect(screen.getByText('Start from Infer')).toBeInTheDocument();
        expect(screen.getByText('Use the command line when you need direct control')).toBeInTheDocument();
        expect(screen.getByText('config.txt')).toBeInTheDocument();
        const codeBlocks = document.querySelectorAll('code');
        expect(codeBlocks[0]).toHaveTextContent('trace serve');
    });

    test('switches between tutorial sections', () => {
        render(<TutorialPanel />);

        fireEvent.click(screen.getByRole('button', { name: /pipeline guide/i }));

        expect(screen.getByText('ML Workflow')).toBeInTheDocument();
        expect(screen.getByText('What this section covers')).toBeInTheDocument();
        expect(screen.getByText('Reviewing metrics and prediction outputs')).toBeInTheDocument();
        expect(screen.queryByText('Web page, server, and command line')).not.toBeInTheDocument();
    });
});
