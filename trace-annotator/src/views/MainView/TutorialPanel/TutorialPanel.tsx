import React, { useState } from 'react';
import './TutorialPanel.scss';

/* ══════════════════════════════════════════════
   SVG FIGURES — inline diagrams for tutorials
   ══════════════════════════════════════════════ */

const C = {
    accent: '#3b9eff',
    cyan: '#58ccff',
    teal: '#54d0c7',
    amber: '#e8a838',
    text1: 'rgba(255,255,255,0.92)',
    text2: 'rgba(255,255,255,0.58)',
    text3: 'rgba(255,255,255,0.36)',
    line: 'rgba(255,255,255,0.12)',
    fill: 'rgba(255,255,255,0.04)',
    fillHi: 'rgba(59,158,255,0.08)',
};

/** Getting Started — campus network with server + recording computers */
const ArchitectureFigure: React.FC = () => (
    <svg className='Figure' viewBox='0 0 640 280' fill='none' role='img' aria-label='Architecture: recording computers on campus network connect to TRACE server via browser'>
        {/* ── Campus network boundary ── */}
        <rect x='6' y='6' width='628' height='268' rx='12' stroke={C.line} strokeWidth='1.2' strokeDasharray='6 4' fill='none' />
        <rect x='232' y='0' width='176' height='16' rx='2' fill='rgba(15,18,30,1)' />
        <text x='320' y='11' textAnchor='middle' fill={C.text3} fontSize='10' fontWeight='600' letterSpacing='0.06em'>CAMPUS / LAB NETWORK</text>

        {/* ── TRACE Server (center) ── */}
        <rect x='236' y='42' width='168' height='168' rx='8' stroke={C.teal} strokeWidth='1.2' fill='rgba(84,208,199,0.05)' />
        <text x='320' y='72' textAnchor='middle' fill={C.text1} fontSize='13' fontWeight='700'>TRACE Server</text>
        <line x1='258' y1='84' x2='382' y2='84' stroke={C.line} />
        <text x='320' y='104' textAnchor='middle' fill={C.text2} fontSize='10'>Web app + API</text>
        <text x='320' y='120' textAnchor='middle' fill={C.text2} fontSize='10'>Shared storage</text>
        <text x='320' y='136' textAnchor='middle' fill={C.text2} fontSize='10'>Job queue</text>
        <text x='320' y='152' textAnchor='middle' fill={C.text2} fontSize='10'>Model outputs</text>
        <text x='320' y='168' textAnchor='middle' fill={C.text2} fontSize='10'>Log streaming</text>
        <text x='320' y='198' textAnchor='middle' fill={C.text3} fontSize='9' fontStyle='italic'>trace serve --host 0.0.0.0</text>

        {/* ── Recording Computer A (left) ── */}
        <rect x='24' y='56' width='150' height='104' rx='8' stroke={C.line} fill={C.fill} />
        <text x='99' y='82' textAnchor='middle' fill={C.text1} fontSize='11' fontWeight='600'>Recording Computer A</text>
        <rect x='44' y='98' width='110' height='28' rx='4' fill={C.fillHi} stroke='rgba(59,158,255,0.2)' />
        <text x='99' y='117' textAnchor='middle' fill={C.cyan} fontSize='11' fontWeight='500'>Browser</text>
        <text x='99' y='148' textAnchor='middle' fill={C.text3} fontSize='9'>e.g. recording room 1</text>

        {/* Arrow A → Server */}
        <line x1='174' y1='110' x2='228' y2='110' stroke={C.accent} strokeWidth='1.5' strokeDasharray='4 3' />
        <polygon points='228,106 236,110 228,114' fill={C.accent} />

        {/* ── Recording Computer B (right) ── */}
        <rect x='466' y='56' width='150' height='104' rx='8' stroke={C.line} fill={C.fill} />
        <text x='541' y='82' textAnchor='middle' fill={C.text1} fontSize='11' fontWeight='600'>Recording Computer B</text>
        <rect x='486' y='98' width='110' height='28' rx='4' fill={C.fillHi} stroke='rgba(59,158,255,0.2)' />
        <text x='541' y='117' textAnchor='middle' fill={C.cyan} fontSize='11' fontWeight='500'>Browser</text>
        <text x='541' y='148' textAnchor='middle' fill={C.text3} fontSize='9'>e.g. recording room 2</text>

        {/* Arrow B → Server */}
        <line x1='466' y1='110' x2='412' y2='110' stroke={C.accent} strokeWidth='1.5' strokeDasharray='4 3' />
        <polygon points='412,106 404,110 412,114' fill={C.accent} />

        {/* ── Recording Computer N (bottom) ── */}
        <rect x='24' y='192' width='150' height='68' rx='8' stroke={C.line} fill={C.fill} />
        <text x='99' y='218' textAnchor='middle' fill={C.text1} fontSize='11' fontWeight='600'>Computer N ...</text>
        <rect x='44' y='230' width='110' height='22' rx='4' fill={C.fillHi} stroke='rgba(59,158,255,0.2)' />
        <text x='99' y='245' textAnchor='middle' fill={C.cyan} fontSize='10' fontWeight='500'>Browser</text>

        {/* Arrow N → Server */}
        <line x1='174' y1='226' x2='236' y2='192' stroke={C.accent} strokeWidth='1.5' strokeDasharray='4 3' />
        <polygon points='231,188 236,192 229,194' fill={C.accent} />

        {/* ── URL hint (bottom-right) ── */}
        <text x='544' y='240' textAnchor='middle' fill={C.text3} fontSize='9.5'>Access at</text>
        <rect x='462' y='248' width='164' height='22' rx='4' fill={C.fillHi} stroke='rgba(59,158,255,0.15)' />
        <text x='544' y='263' textAnchor='middle' fill={C.cyan} fontSize='10' fontFamily='monospace'>http://server:8000</text>
    </svg>
);

/** Getting Started — video intake: two routes */
const VideoIntakeFigure: React.FC = () => (
    <svg className='Figure' viewBox='0 0 600 220' fill='none' role='img' aria-label='Video intake: upload from computer or browse server storage'>
        {/* Route A */}
        <text x='150' y='18' textAnchor='middle' fill={C.accent} fontSize='10' fontWeight='700' letterSpacing='0.08em'>ROUTE A — UPLOAD</text>
        <rect x='28' y='32' width='244' height='56' rx='6' stroke={C.line} fill={C.fill} />
        <text x='150' y='56' textAnchor='middle' fill={C.text1} fontSize='11' fontWeight='600'>Recording computer</text>
        <text x='150' y='72' textAnchor='middle' fill={C.text2} fontSize='10'>Has local video files</text>
        {/* Arrow down */}
        <line x1='150' y1='88' x2='150' y2='114' stroke={C.accent} strokeWidth='1.2' />
        <polygon points='146,114 150,122 154,114' fill={C.accent} />
        <text x='182' y='106' fill={C.text3} fontSize='9'>upload via browser</text>
        {/* Server box */}
        <rect x='58' y='126' width='184' height='48' rx='6' stroke={C.teal} fill='rgba(84,208,199,0.04)' />
        <text x='150' y='148' textAnchor='middle' fill={C.text1} fontSize='11' fontWeight='600'>TRACE server stores file</text>
        <text x='150' y='162' textAnchor='middle' fill={C.text2} fontSize='10'>+ creates browser-ready copy</text>
        {/* Arrow to annotate */}
        <line x1='150' y1='174' x2='150' y2='196' stroke={C.teal} strokeWidth='1.2' />
        <polygon points='146,196 150,204 154,196' fill={C.teal} />
        <text x='150' y='218' textAnchor='middle' fill={C.teal} fontSize='10' fontWeight='600'>Annotate</text>

        {/* Divider */}
        <line x1='300' y1='14' x2='300' y2='218' stroke={C.line} strokeDasharray='3 4' />

        {/* Route B */}
        <text x='450' y='18' textAnchor='middle' fill={C.teal} fontSize='10' fontWeight='700' letterSpacing='0.08em'>ROUTE B — SERVER BROWSE</text>
        <rect x='328' y='32' width='244' height='56' rx='6' stroke={C.teal} fill='rgba(84,208,199,0.04)' />
        <text x='450' y='56' textAnchor='middle' fill={C.text1} fontSize='11' fontWeight='600'>TRACE server</text>
        <text x='450' y='72' textAnchor='middle' fill={C.text2} fontSize='10'>Videos already on shared storage</text>
        {/* Arrow down */}
        <line x1='450' y1='88' x2='450' y2='114' stroke={C.teal} strokeWidth='1.2' />
        <polygon points='446,114 450,122 454,114' fill={C.teal} />
        <text x='496' y='106' fill={C.text3} fontSize='9'>browse from browser</text>
        {/* Direct access */}
        <rect x='358' y='126' width='184' height='48' rx='6' stroke={C.line} fill={C.fill} />
        <text x='450' y='148' textAnchor='middle' fill={C.text1} fontSize='11' fontWeight='600'>Open directly</text>
        <text x='450' y='162' textAnchor='middle' fill={C.text2} fontSize='10'>No upload step needed</text>
        {/* Arrow to annotate */}
        <line x1='450' y1='174' x2='450' y2='196' stroke={C.teal} strokeWidth='1.2' />
        <polygon points='446,196 450,204 454,196' fill={C.teal} />
        <text x='450' y='218' textAnchor='middle' fill={C.teal} fontSize='10' fontWeight='600'>Annotate</text>
    </svg>
);

/** Getting Started — pipeline overview (visual flow) */
const PipelineOverviewFigure: React.FC = () => {
    const stages = [
        { label: 'Prep', color: C.accent, desc: 'Clip videos', out: 'dataset.json' },
        { label: 'Train', color: C.accent, desc: 'Fit model', out: 'best.pth' },
        { label: 'Test', color: C.teal, desc: 'Evaluate', out: 'mAP scores' },
        { label: 'Infer', color: C.amber, desc: 'Predict', out: 'predictions.csv' },
    ];
    const sx = 28, gap = 148, w = 110, h = 80;
    return (
        <svg className='Figure' viewBox='0 0 640 120' fill='none' role='img' aria-label='Pipeline flow: Prep to Train to Test to Infer'>
            {stages.map((s, i) => {
                const x = sx + i * gap;
                return (
                    <React.Fragment key={s.label}>
                        <rect x={x} y='10' width={w} height={h} rx='6' stroke={s.color} strokeWidth='1' fill={`${s.color}10`} />
                        <text x={x + w / 2} y='36' textAnchor='middle' fill={C.text1} fontSize='13' fontWeight='700'>{s.label}</text>
                        <text x={x + w / 2} y='54' textAnchor='middle' fill={C.text2} fontSize='10'>{s.desc}</text>
                        <line x1={x + 14} y1='64' x2={x + w - 14} y2='64' stroke={C.line} />
                        <text x={x + w / 2} y='80' textAnchor='middle' fill={C.text3} fontSize='9.5'>{s.out}</text>
                        {i < stages.length - 1 && (
                            <>
                                <line x1={x + w + 4} y1='50' x2={x + gap - 6} y2='50' stroke={C.line} strokeWidth='1.2' />
                                <polygon points={`${x + gap - 6},46 ${x + gap + 2},50 ${x + gap - 6},54`} fill={C.line} />
                            </>
                        )}
                    </React.Fragment>
                );
            })}
            {/* Input label */}
            <text x={sx + w / 2} y='108' textAnchor='middle' fill={C.text3} fontSize='9'>videos + CSVs</text>
            {/* Output label */}
            <text x={sx + 3 * gap + w / 2} y='108' textAnchor='middle' fill={C.text3} fontSize='9'>JSON output</text>
        </svg>
    );
};

/* ══════════════════════════════════════════════
   DATA TYPES
   ══════════════════════════════════════════════ */

interface TutorialGuideStep {
    id: string;
    step: string;
    title: string;
    description: string;
    bullets?: string[];
    code?: string[];
}

interface TutorialPathSpec {
    id: string;
    title: string;
    description: string;
    items: string[];
    note: string;
}

interface TutorialShortcutGroup {
    category: string;
    items: { keys: string; description: string }[];
}

interface TutorialCliCommand {
    command: string;
    description: string;
    flags?: { flag: string; description: string }[];
    examples?: string[];
}

interface TutorialSection {
    id: string;
    title: string;
    eyebrow: string;
    description: string;
    guideTitle?: string;
    guideSteps?: TutorialGuideStep[];
    pathSpecs?: TutorialPathSpec[];
    shortcuts?: TutorialShortcutGroup[];
    cliCommands?: TutorialCliCommand[];
}

/* ══════════════════════════════════════════════
   SECTION DATA
   ══════════════════════════════════════════════ */

const SECTIONS: TutorialSection[] = [
    {
        id: 'getting-started',
        title: 'Getting Started',
        eyebrow: 'Operator Guide',
        description:
            'Install TRACE on one lab server, then access it from any computer on the campus network via a browser. Each recording computer can upload local videos or annotate directly on shared storage — no local installation needed.',
        guideTitle: 'Quick setup',
        guideSteps: [
            {
                id: 'install',
                step: '01',
                title: 'Install and start the server',
                description: 'Install TRACE on a lab server, then start the server with --host 0.0.0.0 so it listens on the campus network.',
                code: ['pip install trace-tad', 'trace serve --host 0.0.0.0'],
            },
            {
                id: 'open',
                step: '02',
                title: 'Open the web app from any campus computer',
                description: 'The server prints access URLs when it starts. Open the Network URL from any computer on the same network.',
                code: [
                    'Starting TRACE server on http://0.0.0.0:8000',
                    '',
                    '  Local:   http://localhost:8000',
                    '  Network: http://192.168.1.50:8000',
                ],
            },
            {
                id: 'load',
                step: '03',
                title: 'Load videos and start annotating',
                description: 'Choose a server folder or upload local files. TRACE handles codec conversion automatically if the browser cannot play the format.',
            },
        ],
        pathSpecs: [
            {
                id: 'dataset',
                title: 'Dataset folder',
                description: 'Matching videos and CSV annotations. TRACE clips these into training segments automatically.',
                items: ['session01.mp4 + session01.csv', 'session02.mov + session02.csv', 'clips/ (auto-generated)'],
                note: 'If clips/dataset.json already exists, TRACE reuses the prepared dataset.',
            },
            {
                id: 'model',
                title: 'Model folder',
                description: 'Created after training at ./model/. Everything needed to run Test or Infer.',
                items: ['best.pth', 'classmap.txt', 'config.txt'],
                note: 'Pass this folder as --model-path for Test or Infer without retraining.',
            },
            {
                id: 'output',
                title: 'Output artifacts',
                description: 'Test results (mAP scores) are shown directly in the Jobs panel. Inference outputs a CSV you can use for further annotation or downstream processing.',
                items: ['predictions.csv (Infer) — same format as annotation CSVs', 'prep_result.json (Prep) — clip metadata'],
                note: 'Stored in ~/.trace/logs/ alongside job log files.',
            },
        ],
    },
    {
        id: 'annotation',
        title: 'Annotation Guide',
        eyebrow: 'Annotation Workflow',
        description:
            'Create behavior labels on the timeline, navigate quickly with shortcuts, and export CSV annotations aligned with the source videos.',
        shortcuts: [
            {
                category: 'Playback',
                items: [
                    { keys: 'Space', description: 'Play / pause video' },
                    { keys: '\u2190', description: 'Step back one frame' },
                    { keys: '\u2192', description: 'Step forward one frame' },
                ],
            },
            {
                category: 'Navigation',
                items: [
                    { keys: 'Shift + \u2190', description: 'Jump to previous annotation boundary' },
                    { keys: 'Shift + \u2192', description: 'Jump to next annotation boundary' },
                    { keys: 'Ctrl + \u2190', description: 'Reverse-search nearest boundary' },
                    { keys: 'Ctrl + \u2192', description: 'Forward-search nearest boundary' },
                ],
            },
            {
                category: 'Annotation',
                items: [
                    { keys: 'A \u2013 Z', description: 'Toggle behavior assigned to that letter' },
                    { keys: 'Esc', description: 'Clear current recording state' },
                ],
            },
        ],
        guideTitle: 'Step-by-step annotation workflow',
        guideSteps: [
            {
                id: 'load-video',
                step: '01',
                title: 'Load videos into the workspace',
                description: 'From the home screen, choose a server folder or upload local files. Click a video to open it in the editor.',
                bullets: [
                    'Server Folder: type or autocomplete a path on the server.',
                    'Local Upload: drag and drop files from the recording machine.',
                    'HEVC or other unsupported codecs are auto-transcoded to H.264.',
                ],
            },
            {
                id: 'define-labels',
                step: '02',
                title: 'Define behavior labels and assign shortcuts',
                description: 'Open the label editor. Add each behavior, pick a color, and assign a single-letter shortcut key.',
                bullets: [
                    'Labels need unique names (e.g. "grooming", "locomotion", "rearing").',
                    'Shortcut keys are single lowercase letters (a\u2013z).',
                    'Colors are auto-assigned but can be changed.',
                ],
            },
            {
                id: 'annotate',
                step: '03',
                title: 'Mark behavior events on the timeline',
                description: 'Play the video and press a shortcut key when the behavior begins. Press it again when it ends.',
                bullets: [
                    'Space to play/pause. Arrow keys to step frame-by-frame.',
                    'Shift+Arrow jumps between existing annotation boundaries.',
                    'Click the behavior badge bar as an alternative to keyboard shortcuts.',
                ],
            },
            {
                id: 'refine',
                step: '04',
                title: 'Refine clip boundaries',
                description: 'Drag the left or right edge of any clip bar on the timeline to adjust start or end time.',
                bullets: [
                    'Click a clip bar to jump to its start time.',
                    'Delete clips from the clips table using the \u00d7 button.',
                    'Use the minimap for quick navigation on long videos.',
                ],
            },
            {
                id: 'export',
                step: '05',
                title: 'Export annotations',
                description: 'Export as CSV for analysis or to feed into the training pipeline.',
                code: [
                    '# Export format',
                    'image_name,start_frame,end_frame,behavior',
                    'session01.mp4,30,120,grooming',
                    'session01.mp4,150,200,locomotion',
                ],
            },
        ],
    },
    {
        id: 'pipeline',
        title: 'Pipeline Guide',
        eyebrow: 'ML Workflow',
        description:
            'Run the full ML workflow from the browser. Prepare data, train a model, evaluate it, and run inference — all in one pipeline.',
        guideTitle: 'Running a pipeline',
        guideSteps: [
            {
                id: 'select-steps',
                step: '01',
                title: 'Choose which stages to run',
                description: 'Toggle Train, Test, and Infer on or off. The DAG updates to show the active path. Prep runs automatically when Train or Test is enabled.',
            },
            {
                id: 'configure',
                step: '02',
                title: 'Set paths and model size',
                description: 'Fill in the dataset path for Train. It auto-fills into Test and Infer. Choose Small (faster) or Large (higher accuracy).',
                bullets: [
                    'Model Path is only needed when Train is disabled.',
                    'Input Path is for Infer — the video file or folder to predict on.',
                ],
            },
            {
                id: 'run-monitor',
                step: '03',
                title: 'Run and monitor',
                description: 'Click Run Pipeline. Stages execute in order with live log streaming. Cancel at any time.',
                bullets: [
                    'The DAG pulses on the active stage.',
                    'The Jobs panel shows all running and recent jobs.',
                ],
            },
            {
                id: 'review',
                step: '04',
                title: 'Review results',
                description: 'Test shows mAP scores per behavior class. Infer shows predicted segments with labels, timestamps, and confidence scores.',
            },
        ],
        pathSpecs: [
            {
                id: 'dataset',
                title: 'Dataset folder',
                description: 'Videos paired with CSV annotations.',
                items: ['session01.mp4 + session01.csv', 'clips/ (auto-generated)'],
                note: 'CSV format: labelId, timestamp, endTimestamp.',
            },
            {
                id: 'model',
                title: 'Model folder',
                description: 'Created at ./model/ after training.',
                items: ['best.pth', 'classmap.txt', 'config.txt'],
                note: 'Use as --model-path for standalone Test or Infer.',
            },
            {
                id: 'output',
                title: 'Output artifacts',
                description: 'Test mAP scores are shown in the Jobs panel. Inference saves a CSV in annotation format.',
                items: ['predictions.csv (Infer)', 'prep_result.json (Prep)'],
                note: 'Stored in ~/.trace/logs/.',
            },
        ],
    },
    {
        id: 'cli',
        title: 'CLI Guide',
        eyebrow: 'Command Line',
        description:
            'Use the CLI for scripting, remote sessions, or when you prefer direct control over the web UI.',
        guideTitle: 'Running a pipeline from the CLI',
        guideSteps: [
            {
                id: 'cli-train',
                step: '01',
                title: 'Train a model',
                description: 'Point to a dataset folder with videos and CSVs. TRACE auto-clips the videos and prepares annotations before training.',
                code: ['trace train --model large --dataset-path /my/dataset'],
            },
            {
                id: 'cli-test',
                step: '02',
                title: 'Evaluate the model',
                description: 'After training, a model/ folder is created automatically. Use it to evaluate on the same or a different dataset.',
                code: ['trace test --model-path ./model --dataset-path /my/dataset'],
            },
            {
                id: 'cli-infer',
                step: '03',
                title: 'Run inference on new videos',
                description: 'Use the trained model to predict behavior segments on unannotated videos.',
                code: ['trace infer --model-path ./model --input /path/to/video.mp4'],
            },
            {
                id: 'cli-all',
                step: '04',
                title: 'Chain all steps',
                description: 'Run the full pipeline in one line. Each command blocks until it finishes, then the next starts.',
                code: [
                    'trace train --model large --dataset-path /my/dataset && \\',
                    'trace test  --model-path ./model --dataset-path /my/dataset && \\',
                    'trace infer --model-path ./model --input /new/videos/',
                ],
            },
        ],
        cliCommands: [
            {
                command: 'trace serve',
                description: 'Start the web app and API server.',
                flags: [
                    { flag: '--host 0.0.0.0', description: 'Listen on all interfaces (default: localhost)' },
                    { flag: '--port 8080', description: 'Custom port (default: 8000)' },
                    { flag: '--dev', description: 'Dev mode with Vite hot reload on :3000' },
                ],
                examples: ['trace serve', 'trace serve --host 0.0.0.0 --port 8080'],
            },
            {
                command: 'trace train',
                description: 'Train a temporal action detection model.',
                flags: [
                    { flag: '--dataset-path /path', description: 'Folder with videos + CSVs (auto-preps data)' },
                    { flag: '--model small|large', description: 'Model size (default: small)' },
                    { flag: '--nproc N', description: 'Number of GPUs (default: 1)' },
                    { flag: '--config path.py', description: 'Custom config file (overrides --model)' },
                ],
                examples: [
                    'trace train --model small --dataset-path /my/dataset',
                    'trace train --model large --dataset-path /my/dataset --nproc 4',
                ],
            },
            {
                command: 'trace test',
                description: 'Evaluate a trained model on annotated data.',
                flags: [
                    { flag: '--model-path ./model', description: 'Folder with best.pth + classmap.txt' },
                    { flag: '--dataset-path /path', description: 'Test dataset folder' },
                    { flag: '--no-auto-tune', description: 'Disable dataloader auto-tuning' },
                ],
                examples: [
                    'trace test --model-path ./model --dataset-path /my/dataset',
                ],
            },
            {
                command: 'trace infer',
                description: 'Run predictions on new videos.',
                flags: [
                    { flag: '--model-path ./model', description: 'Trained model folder' },
                    { flag: '--input /path', description: 'Video file or folder (required)' },
                    { flag: '--output out.json', description: 'Output JSON path' },
                ],
                examples: [
                    'trace infer --model-path ./model --input /path/to/video.mp4',
                    'trace infer --model-path ./model --input /path/to/videos/',
                ],
            },
        ],
    },
];

/* ══════════════════════════════════════════════
   SECTION ICONS
   ══════════════════════════════════════════════ */

const SectionIcon: React.FC<{ id: string }> = ({ id }) => {
    switch (id) {
        case 'getting-started':
            return (
                <svg width='15' height='15' viewBox='0 0 16 16' fill='none'>
                    <path d='M3 2.5h10M3 8h6M3 13.5h8' stroke='currentColor' strokeWidth='1.3' strokeLinecap='round' />
                </svg>
            );
        case 'annotation':
            return (
                <svg width='15' height='15' viewBox='0 0 16 16' fill='none'>
                    <rect x='1.5' y='1.5' width='13' height='13' rx='2' stroke='currentColor' strokeWidth='1.2' />
                    <path d='M4 8h3M9 8h3M4 5.5h8M4 10.5h5' stroke='currentColor' strokeWidth='1.1' strokeLinecap='round' />
                </svg>
            );
        case 'pipeline':
            return (
                <svg width='15' height='15' viewBox='0 0 16 16' fill='none'>
                    <path d='M2 12l3-5 2.5 3 3-6.5L14 7' stroke='currentColor' strokeWidth='1.3' strokeLinecap='round' strokeLinejoin='round' />
                </svg>
            );
        case 'cli':
            return (
                <svg width='15' height='15' viewBox='0 0 16 16' fill='none'>
                    <rect x='1.5' y='2.5' width='13' height='11' rx='2' stroke='currentColor' strokeWidth='1.2' />
                    <path d='M4.5 6.5L7 8.5 4.5 10.5' stroke='currentColor' strokeWidth='1.2' strokeLinecap='round' strokeLinejoin='round' />
                    <path d='M8.5 10.5h3' stroke='currentColor' strokeWidth='1.2' strokeLinecap='round' />
                </svg>
            );
        default:
            return null;
    }
};

/* ══════════════════════════════════════════════
   MAIN COMPONENT
   ══════════════════════════════════════════════ */

const TutorialPanel: React.FC = () => {
    const [activeId, setActiveId] = useState(SECTIONS[0].id);
    const active = SECTIONS.find((s) => s.id === activeId) ?? SECTIONS[0];

    return (
        <div className='TutorialPanel'>
            <header className='TutorialHeader'>
                <h2 className='TutorialTitle'>Learn TRACE</h2>
                <p className='TutorialSubtitle'>
                    Setup, annotation, pipeline, and CLI reference.
                </p>
            </header>

            <nav className='SectionTabs'>
                {SECTIONS.map((s) => (
                    <button
                        key={s.id}
                        className={`SectionTab ${s.id === activeId ? 'active' : ''}`}
                        onClick={() => setActiveId(s.id)}
                        type='button'
                    >
                        <SectionIcon id={s.id} />
                        <span>{s.title}</span>
                    </button>
                ))}
            </nav>

            <div className='SectionContent' key={activeId}>
                {/* ── Section header ── */}
                <div className='SectionIntro'>
                    <div className='SectionEyebrow'>{active.eyebrow}</div>
                    <h3 className='SectionHeading'>{active.title}</h3>
                    <p className='SectionDesc'>{active.description}</p>
                </div>

                {/* ── Getting Started: figures ── */}
                {active.id === 'getting-started' && (
                    <>
                        <section className='FigureBlock'>
                            <div className='BlockHeader'>
                                <span className='BlockEyebrow'>Shared access</span>
                                <h4 className='BlockTitle'>One server on the campus network, browser access from every computer</h4>
                                <p className='BlockText'>
                                    TRACE runs on a single lab server connected to the campus or lab network. Any recording computer on the same network can access it through a web browser — no software installation needed on client machines.
                                </p>
                            </div>
                            <ArchitectureFigure />
                        </section>

                        <section className='FigureBlock'>
                            <div className='BlockHeader'>
                                <span className='BlockEyebrow'>Video intake</span>
                                <h4 className='BlockTitle'>Two ways to get recordings into TRACE</h4>
                                <p className='BlockText'>
                                    Upload local files through the browser, or browse videos already on server storage. TRACE creates a browser-ready copy if the codec needs conversion.
                                </p>
                            </div>
                            <VideoIntakeFigure />
                        </section>

                        <section className='FigureBlock'>
                            <div className='BlockHeader'>
                                <span className='BlockEyebrow'>Pipeline overview</span>
                                <h4 className='BlockTitle'>Four stages in sequence</h4>
                                <p className='BlockText'>
                                    Each stage produces artifacts for the next. Toggle stages on or off per task — TRACE skips disabled stages and carries outputs forward.
                                </p>
                            </div>
                            <PipelineOverviewFigure />
                        </section>
                    </>
                )}

                {/* ── Pipeline Guide: figure ── */}
                {active.id === 'pipeline' && (
                    <section className='FigureBlock'>
                        <div className='BlockHeader'>
                            <span className='BlockEyebrow'>Execution flow</span>
                            <h4 className='BlockTitle'>Prep → Train → Test → Infer</h4>
                        </div>
                        <PipelineOverviewFigure />
                    </section>
                )}

                {/* ── Keyboard shortcuts ── */}
                {active.shortcuts && (
                    <section className='TutorialBlock'>
                        <div className='BlockHeader'>
                            <span className='BlockEyebrow'>Quick reference</span>
                            <h4 className='BlockTitle'>Keyboard shortcuts</h4>
                        </div>
                        <div className='ShortcutGroups'>
                            {active.shortcuts.map((group) => (
                                <div className='ShortcutGroup' key={group.category}>
                                    <div className='ShortcutCategory'>{group.category}</div>
                                    <div className='ShortcutRows'>
                                        {group.items.map((item) => (
                                            <div className='ShortcutRow' key={item.keys}>
                                                <kbd className='ShortcutKey'>{item.keys}</kbd>
                                                <span className='ShortcutDesc'>{item.description}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </section>
                )}

                {/* ── CLI commands ── */}
                {active.cliCommands && (
                    <section className='TutorialBlock'>
                        <div className='BlockHeader'>
                            <span className='BlockEyebrow'>Commands</span>
                            <h4 className='BlockTitle'>CLI reference</h4>
                        </div>
                        <div className='CliCommands'>
                            {active.cliCommands.map((cmd) => (
                                <div className='CliCommand' key={cmd.command}>
                                    <div className='CliCommandHead'>
                                        <code className='CliCommandName'>{cmd.command}</code>
                                        <span className='CliCommandDesc'>{cmd.description}</span>
                                    </div>
                                    {cmd.flags && (
                                        <div className='CliFlags'>
                                            {cmd.flags.map((f) => (
                                                <div className='CliFlag' key={f.flag}>
                                                    <code className='CliFlagName'>{f.flag}</code>
                                                    <span className='CliFlagDesc'>{f.description}</span>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                    {cmd.examples && (
                                        <pre className='CliExample'>
                                            <code>{cmd.examples.join('\n')}</code>
                                        </pre>
                                    )}
                                </div>
                            ))}
                        </div>
                    </section>
                )}

                {/* ── Guide steps ── */}
                {active.guideSteps && (
                    <section className='TutorialBlock'>
                        <div className='BlockHeader'>
                            <span className='BlockEyebrow'>
                                {active.id === 'annotation' ? 'Walkthrough' : active.id === 'pipeline' ? 'How to' : active.id === 'cli' ? 'Pipeline workflow' : 'Setup'}
                            </span>
                            <h4 className='BlockTitle'>{active.guideTitle ?? 'Guide'}</h4>
                        </div>
                        <div className='GuideSteps'>
                            {active.guideSteps.map((step) => (
                                <article className='GuideStep' key={step.id}>
                                    <div className='GuideIndex'>{step.step}</div>
                                    <div className='GuideBody'>
                                        <h5 className='GuideTitle'>{step.title}</h5>
                                        <p className='GuideText'>{step.description}</p>
                                        {step.bullets && (
                                            <ul className='GuideList'>
                                                {step.bullets.map((b) => <li key={b}>{b}</li>)}
                                            </ul>
                                        )}
                                        {step.code && (
                                            <pre className='GuideCode'><code>{step.code.join('\n')}</code></pre>
                                        )}
                                    </div>
                                </article>
                            ))}
                        </div>
                    </section>
                )}

                {/* ── Path specs ── */}
                {active.pathSpecs && (
                    <section className='TutorialBlock'>
                        <div className='BlockHeader'>
                            <span className='BlockEyebrow'>Path reference</span>
                            <h4 className='BlockTitle'>
                                {active.id === 'pipeline' ? 'Inputs and outputs' : 'Folder conventions'}
                            </h4>
                        </div>
                        <div className='PathGrid'>
                            {active.pathSpecs.map((p) => (
                                <article className='PathCard' key={p.id}>
                                    <h5 className='PathTitle'>{p.title}</h5>
                                    <p className='PathText'>{p.description}</p>
                                    <ul className='PathList'>
                                        {p.items.map((item) => (
                                            <li key={`${p.id}-${item}`}><code>{item}</code></li>
                                        ))}
                                    </ul>
                                    <p className='PathNote'>{p.note}</p>
                                </article>
                            ))}
                        </div>
                    </section>
                )}
            </div>
        </div>
    );
};

export default TutorialPanel;
