import React, { useMemo, useState } from 'react';
import './TutorialPanel.scss';
import { PlatformModel } from '../../../staticModels/PlatformModel';

const formatShortcutKey = (keys: string): string => (
    keys.replace(/Ctrl/g, PlatformModel.isMac ? '⌘ / Ctrl' : 'Ctrl / ⌘')
);

/* ══════════════════════════════════════════════
   SVG FIGURES — inline diagrams for tutorials
   ══════════════════════════════════════════════ */

const C = {
    accent: '#2563eb',
    cyan: '#0891b2',
    teal: '#0d9488',
    amber: '#c2410c',
    text1: 'rgba(15,23,42,0.88)',
    text2: 'rgba(15,23,42,0.62)',
    text3: 'rgba(15,23,42,0.42)',
    line: 'rgba(15,23,42,0.18)',
    fill: 'rgba(15,23,42,0.03)',
    fillHi: 'rgba(37,99,235,0.06)',
};

/** Getting Started — campus network with server + recording computers */
const ArchitectureFigure: React.FC = () => (
    <svg className='Figure' viewBox='0 0 640 280' fill='none' role='img' aria-label='Architecture: recording computers on campus network connect to TRACE server via browser'>
        <rect x='6' y='6' width='628' height='268' rx='12' stroke={C.line} strokeWidth='1.2' strokeDasharray='6 4' fill='none' />
        <rect x='232' y='0' width='176' height='16' rx='2' fill='#ffffff' stroke={C.line} strokeWidth='1' />
        <text x='320' y='11' textAnchor='middle' fill={C.text2} fontSize='11' fontWeight='700' letterSpacing='0.16em'>CAMPUS / LAB NETWORK</text>

        <rect x='236' y='42' width='168' height='168' rx='8' stroke={C.teal} strokeWidth='1.2' fill='rgba(84,208,199,0.05)' />
        <text x='320' y='72' textAnchor='middle' fill={C.text1} fontSize='15' fontWeight='700'>TRACE Server</text>
        <line x1='258' y1='84' x2='382' y2='84' stroke={C.line} />
        <text x='320' y='104' textAnchor='middle' fill={C.text2} fontSize='12'>Web app + API</text>
        <text x='320' y='120' textAnchor='middle' fill={C.text2} fontSize='12'>Shared storage</text>
        <text x='320' y='136' textAnchor='middle' fill={C.text2} fontSize='12'>Job queue</text>
        <text x='320' y='152' textAnchor='middle' fill={C.text2} fontSize='12'>Model outputs</text>
        <text x='320' y='168' textAnchor='middle' fill={C.text2} fontSize='12'>Log streaming</text>
        <text x='320' y='198' textAnchor='middle' fill={C.text3} fontSize='11' fontStyle='italic'>trace app --host 0.0.0.0</text>

        <rect x='24' y='56' width='150' height='104' rx='8' stroke={C.line} fill={C.fill} />
        <text x='99' y='82' textAnchor='middle' fill={C.text1} fontSize='13' fontWeight='600'>Recording Computer A</text>
        <rect x='44' y='98' width='110' height='28' rx='4' fill={C.fillHi} stroke='rgba(59,158,255,0.2)' />
        <text x='99' y='117' textAnchor='middle' fill={C.cyan} fontSize='13' fontWeight='600'>Browser</text>
        <text x='99' y='148' textAnchor='middle' fill={C.text3} fontSize='11'>e.g. recording room 1</text>

        <line x1='174' y1='110' x2='228' y2='110' stroke={C.accent} strokeWidth='1.5' strokeDasharray='4 3' />
        <polygon points='228,106 236,110 228,114' fill={C.accent} />

        <rect x='466' y='56' width='150' height='104' rx='8' stroke={C.line} fill={C.fill} />
        <text x='541' y='82' textAnchor='middle' fill={C.text1} fontSize='13' fontWeight='600'>Recording Computer B</text>
        <rect x='486' y='98' width='110' height='28' rx='4' fill={C.fillHi} stroke='rgba(59,158,255,0.2)' />
        <text x='541' y='117' textAnchor='middle' fill={C.cyan} fontSize='13' fontWeight='600'>Browser</text>
        <text x='541' y='148' textAnchor='middle' fill={C.text3} fontSize='11'>e.g. recording room 2</text>

        <line x1='466' y1='110' x2='412' y2='110' stroke={C.accent} strokeWidth='1.5' strokeDasharray='4 3' />
        <polygon points='412,106 404,110 412,114' fill={C.accent} />

        <rect x='24' y='192' width='150' height='68' rx='8' stroke={C.line} fill={C.fill} />
        <text x='99' y='218' textAnchor='middle' fill={C.text1} fontSize='13' fontWeight='600'>Computer N ...</text>
        <rect x='44' y='230' width='110' height='22' rx='4' fill={C.fillHi} stroke='rgba(59,158,255,0.2)' />
        <text x='99' y='245' textAnchor='middle' fill={C.cyan} fontSize='12' fontWeight='600'>Browser</text>

        <line x1='174' y1='226' x2='236' y2='192' stroke={C.accent} strokeWidth='1.5' strokeDasharray='4 3' />
        <polygon points='231,188 236,192 229,194' fill={C.accent} />

        <text x='544' y='240' textAnchor='middle' fill={C.text3} fontSize='11'>Access at</text>
        <rect x='462' y='248' width='164' height='22' rx='4' fill={C.fillHi} stroke='rgba(59,158,255,0.15)' />
        <text x='544' y='263' textAnchor='middle' fill={C.cyan} fontSize='12' fontFamily='monospace'>http://server:8000</text>
    </svg>
);

/** Getting Started — video intake: two routes */
const VideoIntakeFigure: React.FC = () => (
    <svg className='Figure' viewBox='0 0 600 220' fill='none' role='img' aria-label='Video intake: upload from computer or browse server storage'>
        <text x='150' y='18' textAnchor='middle' fill={C.accent} fontSize='11' fontWeight='700' letterSpacing='0.16em'>ROUTE A: UPLOAD</text>
        <rect x='28' y='32' width='244' height='56' rx='6' stroke={C.line} fill={C.fill} />
        <text x='150' y='56' textAnchor='middle' fill={C.text1} fontSize='13' fontWeight='600'>Recording computer</text>
        <text x='150' y='72' textAnchor='middle' fill={C.text2} fontSize='12'>Has local video files</text>
        <line x1='150' y1='88' x2='150' y2='114' stroke={C.accent} strokeWidth='1.2' />
        <polygon points='146,114 150,122 154,114' fill={C.accent} />
        <text x='182' y='106' fill={C.text3} fontSize='11'>upload via browser</text>
        <rect x='58' y='126' width='184' height='48' rx='6' stroke={C.teal} fill='rgba(84,208,199,0.04)' />
        <text x='150' y='148' textAnchor='middle' fill={C.text1} fontSize='13' fontWeight='600'>TRACE server stores file</text>
        <text x='150' y='162' textAnchor='middle' fill={C.text2} fontSize='12'>+ creates browser-ready copy</text>
        <line x1='150' y1='174' x2='150' y2='196' stroke={C.teal} strokeWidth='1.2' />
        <polygon points='146,196 150,204 154,196' fill={C.teal} />
        <text x='150' y='218' textAnchor='middle' fill={C.teal} fontSize='12' fontWeight='600'>Annotate</text>

        <line x1='300' y1='14' x2='300' y2='218' stroke={C.line} strokeDasharray='3 4' />

        <text x='450' y='18' textAnchor='middle' fill={C.teal} fontSize='11' fontWeight='700' letterSpacing='0.16em'>ROUTE B: SERVER BROWSE</text>
        <rect x='328' y='32' width='244' height='56' rx='6' stroke={C.teal} fill='rgba(84,208,199,0.04)' />
        <text x='450' y='56' textAnchor='middle' fill={C.text1} fontSize='13' fontWeight='600'>TRACE server</text>
        <text x='450' y='72' textAnchor='middle' fill={C.text2} fontSize='12'>Videos already on shared storage</text>
        <line x1='450' y1='88' x2='450' y2='114' stroke={C.teal} strokeWidth='1.2' />
        <polygon points='446,114 450,122 454,114' fill={C.teal} />
        <text x='404' y='106' textAnchor='end' fill={C.text3} fontSize='11'>browse from browser</text>
        <rect x='358' y='126' width='184' height='48' rx='6' stroke={C.line} fill={C.fill} />
        <text x='450' y='148' textAnchor='middle' fill={C.text1} fontSize='13' fontWeight='600'>Open directly</text>
        <text x='450' y='162' textAnchor='middle' fill={C.text2} fontSize='12'>No upload step needed</text>
        <line x1='450' y1='174' x2='450' y2='196' stroke={C.teal} strokeWidth='1.2' />
        <polygon points='446,196 450,204 454,196' fill={C.teal} />
        <text x='450' y='218' textAnchor='middle' fill={C.teal} fontSize='12' fontWeight='600'>Annotate</text>
    </svg>
);

/** Getting Started: Run Model overview (visual flow) */
const PipelineOverviewFigure: React.FC = () => {
    const stages = [
        { label: 'Prep', color: C.accent, desc: 'Clip videos', out: 'dataset.json' },
        { label: 'Train', color: C.accent, desc: 'Fit model', out: 'best.pth' },
        { label: 'Test', color: C.teal, desc: 'Evaluate', out: 'mAP scores' },
        { label: 'Predict', color: C.amber, desc: 'New videos', out: 'predictions.csv' },
    ];
    const sx = 28, gap = 148, w = 110, h = 80;
    return (
        <svg className='Figure' viewBox='0 0 640 120' fill='none' role='img' aria-label='Run Model flow: Prep to Train to Test to Predict'>
            {stages.map((s, i) => {
                const x = sx + i * gap;
                return (
                    <React.Fragment key={s.label}>
                        <rect x={x} y='10' width={w} height={h} rx='6' stroke={s.color} strokeWidth='1' fill={`${s.color}10`} />
                        <text x={x + w / 2} y='36' textAnchor='middle' fill={C.text1} fontSize='15' fontWeight='700'>{s.label}</text>
                        <text x={x + w / 2} y='54' textAnchor='middle' fill={C.text2} fontSize='12'>{s.desc}</text>
                        <line x1={x + 14} y1='64' x2={x + w - 14} y2='64' stroke={C.line} />
                        <text x={x + w / 2} y='80' textAnchor='middle' fill={C.text3} fontSize='11'>{s.out}</text>
                        {i < stages.length - 1 && (
                            <>
                                <line x1={x + w + 4} y1='50' x2={x + gap - 6} y2='50' stroke={C.line} strokeWidth='1.2' />
                                <polygon points={`${x + gap - 6},46 ${x + gap + 2},50 ${x + gap - 6},54`} fill={C.line} />
                            </>
                        )}
                    </React.Fragment>
                );
            })}
            <text x={sx + w / 2} y='108' textAnchor='middle' fill={C.text3} fontSize='11'>videos + CSVs</text>
            <text x={sx + 3 * gap + w / 2} y='108' textAnchor='middle' fill={C.text3} fontSize='11'>JSON output</text>
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

interface TutorialPhilosophyPoint {
    title: string;
    text: string;
}

interface TutorialSection {
    id: string;
    title: string;
    eyebrow: string;
    description: string;
    philosophy?: TutorialPhilosophyPoint[];
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
            'TRACE keeps lab video work local and reproducible: pair videos with annotation files, train a model, then use it to draft annotations for new recordings.',
        philosophy: [
            {
                title: 'Pair videos with annotations',
                text: 'One recording can keep multiple CSVs, such as drafts, final labels, or separate raters.',
            },
            {
                title: 'Train from selected pairs',
                text: 'Choose the video and annotation pairs for a run; the model keeps its config, classes, and checkpoint.',
            },
            {
                title: 'Predict new annotation files',
                text: 'Run the trained model on new videos to create annotation CSVs ready for review.',
            },
        ],
        guideTitle: 'Quick setup',
        guideSteps: [
            {
                id: 'install',
                step: '01',
                title: 'Install and start the server',
                description: 'Install TRACE on a lab server. Start it on the campus network when others need browser access.',
                code: ['pip install trace-tad', 'trace app --host 0.0.0.0'],
            },
            {
                id: 'open',
                step: '02',
                title: 'Open the web app from any campus computer',
                description: 'Use the Network URL printed by the server.',
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
                description: 'Choose a server folder or upload local files. TRACE converts browser-unfriendly codecs when needed.',
            },
        ],
        pathSpecs: [
            {
                id: 'dataset',
                title: 'Dataset folder',
                description: 'Source videos and the annotation CSVs that belong to them.',
                items: ['session01.mp4 + session01.csv', 'session01.mp4 + session01_reviewerA.csv', 'model_YYYYMMDD_HHMMSS/ (auto-generated)'],
                note: 'Training builds dataset.json and classmap.txt from the selected video/annotation pairs.',
            },
            {
                id: 'model',
                title: 'Model folder',
                description: 'The reusable model bundle created by training.',
                items: ['best.pth', 'dataset.json', 'classmap.txt', 'config.txt'],
                note: 'Use it for Test or Predict without retraining.',
            },
            {
                id: 'output',
                title: 'Output artifacts',
                description: 'Files produced by Prep, Test, and Predict.',
                items: ['predictions.csv (Predict), same format as annotation CSVs', 'prep_result.json (Prep), dataset metadata'],
                note: 'Predict writes annotation drafts beside the selected input videos.',
            },
        ],
    },
    {
        id: 'annotation',
        title: 'Annotation Guide',
        eyebrow: 'Annotation Workflow',
        description:
            'Label behavior on the timeline and save plain CSVs that stay aligned with each source video.',
        shortcuts: [
            {
                category: 'Playback',
                items: [
                    { keys: 'Space', description: 'Play / pause video' },
                    { keys: '←', description: 'Step back one frame' },
                    { keys: '→', description: 'Step forward one frame' },
                ],
            },
            {
                category: 'Navigation',
                items: [
                    { keys: 'Shift + ←', description: 'Jump to previous annotation boundary' },
                    { keys: 'Shift + →', description: 'Jump to next annotation boundary' },
                    { keys: 'Ctrl + ←', description: 'Snap next boundary to current frame' },
                    { keys: 'Ctrl + →', description: 'Snap previous boundary to current frame' },
                ],
            },
            {
                category: 'Annotation',
                items: [
                    { keys: 'A-Z', description: 'Toggle behavior assigned to that letter' },
                    { keys: 'Delete', description: 'Delete the active behavior clip' },
                    { keys: 'Esc', description: 'Clear current recording state' },
                ],
            },
        ],
        guideTitle: 'Annotation flow',
        guideSteps: [
            {
                id: 'load-video',
                step: '01',
                title: 'Load videos into the workspace',
                description: 'Choose a server folder or upload local files. Click a video to open it.',
                bullets: [
                    'Server Folder opens files already on shared storage.',
                    'Local Upload copies files from the recording machine.',
                    'Convert browser-unfriendly codecs before or during review.',
                ],
            },
            {
                id: 'define-labels',
                step: '02',
                title: 'Define behavior labels and assign shortcuts',
                description: 'Add behaviors, choose colors, and assign single-letter keys.',
                bullets: [
                    'Use unique names such as grooming, locomotion, or rearing.',
                    'Shortcuts use one lowercase letter.',
                    'Colors can be changed later.',
                ],
            },
            {
                id: 'annotate',
                step: '03',
                title: 'Mark behavior events on the timeline',
                description: 'Press a shortcut when behavior starts. Press it again when it ends.',
                bullets: [
                    'Space to play/pause. Arrow keys to step frame-by-frame.',
                    'Shift+Arrow jumps between existing annotation boundaries.',
                    'Click behavior badges if you prefer the mouse.',
                ],
            },
            {
                id: 'refine',
                step: '04',
                title: 'Refine clip boundaries',
                description: 'Drag clip edges to adjust start and end times.',
                bullets: [
                    'Click a clip bar to jump to its start time.',
                    'Delete mistakes from the clips table.',
                    'Use the minimap on long recordings.',
                ],
            },
            {
                id: 'export',
                step: '05',
                title: 'Export annotations',
                description: 'Export CSVs for analysis or model training.',
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
        title: 'Run Model Guide',
        eyebrow: 'ML Workflow',
        description:
            'Prepare data, train, test, and predict from the browser. Each stage writes files the next stage can use.',
        guideTitle: 'Model run',
        guideSteps: [
            {
                id: 'select-steps',
                step: '01',
                title: 'Choose which stages to run',
                description: 'Toggle Train, Test, and Predict. Prep runs automatically when needed.',
            },
            {
                id: 'configure',
                step: '02',
                title: 'Set paths and model size',
                description: 'Choose data paths and model size. Small is faster; Large can be more accurate.',
                bullets: [
                    'Model Path is only needed when Train is disabled.',
                    'Input Path is the video file or folder for Predict.',
                ],
            },
            {
                id: 'run-monitor',
                step: '03',
                title: 'Run and monitor',
                description: 'Click Run Model. Watch live logs and cancel if needed.',
                bullets: [
                    'The DAG pulses on the active stage.',
                    'The Jobs panel keeps recent runs visible.',
                ],
            },
            {
                id: 'review',
                step: '04',
                title: 'Review results',
                description: 'Test shows mAP by behavior. Predict shows labels, times, and scores.',
            },
        ],
        pathSpecs: [
            {
                id: 'dataset',
                title: 'Dataset folder',
                description: 'Videos paired with annotation CSVs.',
                items: ['session01.mp4 + session01.csv', 'model_YYYYMMDD_HHMMSS/ (auto-generated)'],
                note: 'CSV format: labelId, timestamp, endTimestamp.',
            },
            {
                id: 'model',
                title: 'Model folder',
                description: 'Created after training.',
                items: ['best.pth', 'dataset.json', 'classmap.txt', 'config.txt'],
                note: 'Use as --model-dir for standalone Test or Predict.',
            },
            {
                id: 'output',
                title: 'Output artifacts',
                description: 'Test metrics and prediction files.',
                items: ['predictions.csv (Predict)', 'prep_result.json (Prep)'],
                note: 'Predict writes beside the selected input videos.',
            },
        ],
    },
    {
        id: 'cli',
        title: 'CLI Guide',
        eyebrow: 'Command Line',
        description:
            'Use the CLI for scripts, remote sessions, and repeatable runs.',
        guideTitle: 'CLI workflow',
        guideSteps: [
            {
                id: 'cli-train',
                step: '01',
                title: 'Train a model',
                description: 'Point to paired videos and CSVs. TRACE prepares metadata before training.',
                code: ['trace train --model large --work-dir /my/dataset --pairs session01.mp4=session01.csv'],
            },
            {
                id: 'cli-test',
                step: '02',
                title: 'Evaluate the model',
                description: 'Use the model timestamp folder to evaluate the same or a new dataset.',
                code: ['trace eval --model-dir /my/dataset/model_YYYYMMDD_HHMMSS'],
            },
            {
                id: 'cli-infer',
                step: '03',
                title: 'Run inference on new videos',
                description: 'Predict behavior segments on unannotated videos.',
                code: ['trace predict --model-dir /my/dataset/model_YYYYMMDD_HHMMSS --input /path/to/video.mp4'],
            },
            {
                id: 'cli-all',
                step: '04',
                title: 'Chain all steps',
                description: 'Run each command in order when you need a scripted workflow.',
                code: [
                    'trace train --model large --work-dir /my/dataset --pairs session01.mp4=session01.csv',
                    'trace eval --model-dir /my/dataset/model_YYYYMMDD_HHMMSS',
                    'trace predict --model-dir /my/dataset/model_YYYYMMDD_HHMMSS --input /new/videos/',
                ],
            },
        ],
        cliCommands: [
            {
                command: 'trace update',
                description: 'Check PyPI for a newer TRACE package.',
                flags: [
                    { flag: '--timeout N', description: 'Seconds to wait for PyPI (default: 5)' },
                ],
                examples: ['trace update'],
            },
            {
                command: 'trace app',
                description: 'Start the web app and API.',
                flags: [
                    { flag: '--host 0.0.0.0', description: 'Listen on the lab network' },
                    { flag: '--port 8080', description: 'Custom port (default: 8000)' },
                    { flag: '--dev', description: 'Use frontend hot reload' },
                ],
                examples: ['trace app', 'trace app --host 0.0.0.0 --port 8080'],
            },
            {
                command: 'trace train',
                description: 'Train a behavior detector.',
                flags: [
                    { flag: '--work-dir /path', description: 'Folder with videos and annotation CSVs' },
                    { flag: '--pairs video=csv', description: 'Selected video/annotation pairs' },
                    { flag: '--model small|large', description: 'Model size (default: small)' },
                    { flag: '--nproc N', description: 'Number of GPUs (default: 1)' },
                    { flag: '--config path.py', description: 'Custom config file (overrides --model)' },
                ],
                examples: [
                    'trace train --model small --work-dir /my/dataset --pairs session01.mp4=session01.csv',
                    'trace train --model large --work-dir /my/dataset --pairs session01.mp4=session01.csv --nproc 4',
                ],
            },
            {
                command: 'trace eval',
                description: 'Score a model on annotated data.',
                flags: [
                    { flag: '--model-dir /path/model_...', description: 'Model artifact folder' },
                    { flag: '--work-dir /path', description: 'Optional test dataset folder' },
                    { flag: '--pairs video=csv', description: 'Explicit test pairs when --work-dir is set' },
                    { flag: '--auto-tune', description: 'Benchmark dataloader settings' },
                ],
                examples: [
                    'trace eval --model-dir /my/dataset/model_YYYYMMDD_HHMMSS',
                ],
            },
            {
                command: 'trace predict',
                description: 'Predict on new videos.',
                flags: [
                    { flag: '--model-dir /path/model_...', description: 'Trained model folder' },
                    { flag: '--input /path', description: 'Video file or folder' },
                    { flag: '--output out.json', description: 'Output JSON path' },
                    { flag: '--auto-tune', description: 'Benchmark dataloader settings' },
                ],
                examples: [
                    'trace predict --model-dir /my/dataset/model_YYYYMMDD_HHMMSS --input /path/to/video.mp4',
                    'trace predict --model-dir /my/dataset/model_YYYYMMDD_HHMMSS --input /path/to/videos/',
                ],
            },
        ],
    },
];

/* ══════════════════════════════════════════════
   CHAPTER NUMBER (display only)
   ══════════════════════════════════════════════ */

const sectionNum = (idx: number) => String(idx + 1).padStart(2, '0');

/* ══════════════════════════════════════════════
   ANCHOR LIST — built from each section's blocks
   ══════════════════════════════════════════════ */

const sectionAnchors = (s: TutorialSection): { id: string; label: string }[] => {
    const list: { id: string; label: string }[] = [{ id: 'overview', label: 'Overview' }];
    if (s.philosophy) list.push({ id: 'philosophy', label: 'Philosophy' });
    if (s.id === 'getting-started') {
        list.push(
            { id: 'architecture', label: 'Architecture' },
            { id: 'video-intake', label: 'Video intake' },
            { id: 'pipeline-overview', label: 'Run Model overview' },
        );
    }
    if (s.id === 'pipeline') {
        list.push({ id: 'flow', label: 'Execution flow' });
    }
    if (s.shortcuts) list.push({ id: 'shortcuts', label: 'Shortcuts' });
    if (s.cliCommands) list.push({ id: 'cli-ref', label: 'Commands' });
    if (s.guideSteps) list.push({ id: 'walkthrough', label: 'Walkthrough' });
    if (s.pathSpecs) list.push({ id: 'paths', label: 'Paths' });
    return list;
};

/* ══════════════════════════════════════════════
   MAIN COMPONENT — two-pane manual layout
   ══════════════════════════════════════════════ */

const TutorialPanel: React.FC = () => {
    const [activeId, setActiveId] = useState(SECTIONS[0].id);
    const active = SECTIONS.find((s) => s.id === activeId) ?? SECTIONS[0];
    const activeIdx = SECTIONS.findIndex((s) => s.id === activeId);
    const anchors = useMemo(() => sectionAnchors(active), [active.id]);

    return (
        <div className='TutorialPanel'>
            {/* ── Title bar ── */}
            <header className='TutorialHeader'>
                <div className='THeadLeft'>
                    <span className='THeadMark'>§ MANUAL</span>
                    <h2 className='TutorialTitle'>Learn TRACE</h2>
                </div>
                <p className='TutorialSubtitle'>
                    Philosophy, setup, annotation, models, and CLI.
                </p>
            </header>

            {/* ── Two-pane manual layout ── */}
            <div className='TutorialBody'>
                {/* Left rail: sections + anchors */}
                <aside className='TutorialRail' aria-label='Tutorial sections'>
                    <ol className='RailSections'>
                        {SECTIONS.map((s, idx) => {
                            const isActive = s.id === activeId;
                            return (
                                <li key={s.id} className={`RailSection ${isActive ? 'active' : ''}`}>
                                    <button
                                        type='button'
                                        className='RailSectionBtn'
                                        onClick={() => setActiveId(s.id)}
                                        aria-current={isActive ? 'page' : undefined}
                                        aria-label={`${s.title} guide`}
                                    >
                                        <span className='RailNum'>§ {sectionNum(idx)}</span>
                                        <span className='RailTitle'>{s.title}</span>
                                    </button>
                                    {isActive && anchors.length > 1 && (
                                        <ol className='RailAnchors'>
                                            {anchors.map((a) => (
                                                <li key={a.id}>
                                                    <a className='RailAnchor' href={`#${active.id}-${a.id}`}>
                                                        <span className='RailAnchorBullet' aria-hidden />
                                                        <span>{a.label}</span>
                                                    </a>
                                                </li>
                                            ))}
                                        </ol>
                                    )}
                                </li>
                            );
                        })}
                    </ol>
                </aside>

                {/* Right pane: flowing content */}
                <main className='TutorialContent' key={activeId}>
                    {/* Section banner */}
                    <section id={`${active.id}-overview`} className='TutorialBanner'>
                        <div className='BannerHead'>
                            <span className='BannerNum'>§ {sectionNum(activeIdx)}</span>
                            <span className='BannerEyebrow'>{active.eyebrow}</span>
                        </div>
                        <h3 className='BannerTitle'>{active.title}</h3>
                        <p className='BannerDesc'>{active.description}</p>
                    </section>

                    {active.philosophy && (
                        <section id={`${active.id}-philosophy`} className='TBlock'>
                            <BlockHead num='00' eyebrow='TRACE workflow' title='From video pairs to predicted annotations' />
                            <div className='PhilosophyGrid'>
                                {active.philosophy.map((point, i) => (
                                    <div className='PhilosophyPoint' key={point.title}>
                                        <span className='PhilosophyMark'>· {String(i + 1).padStart(2, '0')}</span>
                                        <h5 className='PhilosophyTitle'>{point.title}</h5>
                                        <p className='PhilosophyText'>{point.text}</p>
                                    </div>
                                ))}
                            </div>
                        </section>
                    )}

                    {/* Getting Started — figures */}
                    {active.id === 'getting-started' && (
                        <>
                            <section id='getting-started-architecture' className='TBlock'>
                                <BlockHead num='01' eyebrow='Shared access' title='One server, browser access'>
                                    Run TRACE on a lab server. Recording computers open it in a browser on the same network.
                                </BlockHead>
                                <ArchitectureFigure />
                            </section>

                            <section id='getting-started-video-intake' className='TBlock'>
                                <BlockHead num='02' eyebrow='Video intake' title='Two ways to open recordings'>
                                    Upload local files or browse server storage. TRACE creates a browser-ready copy when needed.
                                </BlockHead>
                                <VideoIntakeFigure />
                            </section>

                            <section id='getting-started-pipeline-overview' className='TBlock'>
                                <BlockHead num='03' eyebrow='Run Model overview' title='Four stages in sequence'>
                                    Prep, Train, Test, and Predict can run together or as separate steps.
                                </BlockHead>
                                <PipelineOverviewFigure />
                            </section>
                        </>
                    )}

                    {/* Run Model guide figure */}
                    {active.id === 'pipeline' && (
                        <section id='pipeline-flow' className='TBlock'>
                            <BlockHead num='00' eyebrow='Execution flow' title='Prep → Train → Test → Predict' />
                            <PipelineOverviewFigure />
                        </section>
                    )}

                    {/* Keyboard shortcuts — flat 3-column grid, no cards */}
                    {active.shortcuts && (
                        <section id={`${active.id}-shortcuts`} className='TBlock'>
                            <BlockHead num='REF' eyebrow='Quick reference' title='Keyboard shortcuts' />
                            <div className='ShortcutGrid'>
                                {active.shortcuts.map((group) => (
                                    <div className='ShortcutCol' key={group.category}>
                                        <div className='ShortcutCat'>{group.category}</div>
                                        {group.items.map((item) => (
                                            <div className='ShortcutRow' key={item.keys}>
                                                <kbd className='Kbd'>{formatShortcutKey(item.keys)}</kbd>
                                                <span className='KbdDesc'>{item.description}</span>
                                            </div>
                                        ))}
                                    </div>
                                ))}
                            </div>
                        </section>
                    )}

                    {/* CLI commands — full-width tabular layout */}
                    {active.cliCommands && (
                        <section id={`${active.id}-cli-ref`} className='TBlock'>
                            <BlockHead num='REF' eyebrow='Commands' title='CLI reference' />
                            <div className='CliList'>
                                {active.cliCommands.map((cmd) => (
                                    <article className='CliEntry' key={cmd.command}>
                                        <header className='CliEntryHead'>
                                            <code className='CliEntryName'>{cmd.command}</code>
                                            <span className='CliEntryDesc'>{cmd.description}</span>
                                        </header>
                                        <div className='CliEntryBody'>
                                            {cmd.flags && (
                                                <dl className='CliFlagList'>
                                                    {cmd.flags.map((f) => (
                                                        <React.Fragment key={f.flag}>
                                                            <dt><code>{f.flag}</code></dt>
                                                            <dd>{f.description}</dd>
                                                        </React.Fragment>
                                                    ))}
                                                </dl>
                                            )}
                                            {cmd.examples && (
                                                <pre className='CliExample'>
                                                    <code>{cmd.examples.map((ex) => `$ ${ex}`).join('\n')}</code>
                                                </pre>
                                            )}
                                        </div>
                                    </article>
                                ))}
                            </div>
                        </section>
                    )}

                    {/* Guide steps — flat numbered list, no cards */}
                    {active.guideSteps && (
                        <section id={`${active.id}-walkthrough`} className='TBlock'>
                            <BlockHead
                                num='REF'
                                eyebrow={active.id === 'annotation' ? 'Walkthrough' : active.id === 'pipeline' ? 'How to' : active.id === 'cli' ? 'Pipeline workflow' : 'Setup'}
                                title={active.guideTitle ?? 'Guide'}
                            />
                            <ol className='GuideSteps'>
                                {active.guideSteps.map((step) => (
                                    <li className='GuideStep' key={step.id}>
                                        <div className='GuideRail'>
                                            <span className='GuideStepNum'>{step.step}</span>
                                            <span className='GuideStepLine' aria-hidden />
                                        </div>
                                        <div className='GuideStepBody'>
                                            <h5 className='GuideStepTitle'>{step.title}</h5>
                                            <p className='GuideStepText'>{step.description}</p>
                                            {step.bullets && (
                                                <ul className='GuideStepList'>
                                                    {step.bullets.map((b) => <li key={b}>{b}</li>)}
                                                </ul>
                                            )}
                                            {step.code && (
                                                <pre className='GuideStepCode'><code>{step.code.join('\n')}</code></pre>
                                            )}
                                        </div>
                                    </li>
                                ))}
                            </ol>
                        </section>
                    )}

                    {/* Path specs — 3-up flat columns, no card backgrounds */}
                    {active.pathSpecs && (
                        <section id={`${active.id}-paths`} className='TBlock'>
                            <BlockHead
                                num='REF'
                                eyebrow='Path reference'
                                title={active.id === 'pipeline' ? 'Inputs and outputs' : 'Folder conventions'}
                            />
                            <div className='PathGrid'>
                                {active.pathSpecs.map((p, i) => (
                                    <div className='PathCol' key={p.id}>
                                        <span className='PathColMark'>· {String(i + 1).padStart(2, '0')}</span>
                                        <h5 className='PathColTitle'>{p.title}</h5>
                                        <p className='PathColText'>{p.description}</p>
                                        <ul className='PathColList'>
                                            {p.items.map((item) => (
                                                <li key={`${p.id}-${item}`}><code>{item}</code></li>
                                            ))}
                                        </ul>
                                        <p className='PathColNote'>{p.note}</p>
                                    </div>
                                ))}
                            </div>
                        </section>
                    )}
                </main>
            </div>
        </div>
    );
};

/* Shared block header — § eyebrow + title row */
const BlockHead: React.FC<{ num: string; eyebrow: string; title: string; children?: React.ReactNode }> = ({
    num, eyebrow, title, children,
}) => (
    <header className='BlockHead'>
        <div className='BlockHeadRow'>
            <span className='BlockHeadNum'>§ {num}</span>
            <span className='BlockHeadEyebrow'>{eyebrow}</span>
            <span className='BlockHeadRule' aria-hidden />
        </div>
        <h4 className='BlockHeadTitle'>{title}</h4>
        {children && <p className='BlockHeadText'>{children}</p>}
    </header>
);

export default TutorialPanel;
