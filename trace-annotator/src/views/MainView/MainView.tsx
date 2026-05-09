import React from 'react';
import './MainView.scss';
import ImagesDropZone from './ImagesDropZone/ImagesDropZone';
import PipelineBuilder from './PipelineBuilder/PipelineBuilder';
import TutorialPanel from './TutorialPanel/TutorialPanel';
import { usePipelineState } from './PipelineBuilder/usePipelineState';
import { connect } from 'react-redux';
import { AppState } from '../../store';
import { updateHomeTab } from '../../store/general/actionCreators';
import { HomeTab } from '../../store/general/types';

const AcronymWord: React.FC<{ word: string; highlightIndex: number }> = ({ word, highlightIndex }) => (
    <span className='AWord'>
        {word.split('').map((ch, i) => (
            <span key={i} className={i === highlightIndex ? 'hi' : undefined}>{ch}</span>
        ))}
    </span>
);

interface IProps {
    homeTab: HomeTab;
    updateHomeTab: (tab: HomeTab) => any;
}

const MainView: React.FC<IProps> = ({ homeTab: mode, updateHomeTab: setMode }) => {
    const pipeline = usePipelineState();

    return (
        <div className='MainView'>
            <div className='BgGrid' />
            <div className='BgGlow' />

            <div className='MainShell'>
                {/* Top: Brand banner */}
                <header className='BrandBanner'>
                    <div className='BrandRow'>
                        <div className='LogoMark'>
                            <svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M11 5 L6 5 L6 27 L11 27"
                                    stroke="currentColor" strokeWidth="2"
                                    strokeLinecap="round" strokeLinejoin="round"
                                    opacity="0.75"/>
                                <path d="M21 5 L26 5 L26 27 L21 27"
                                    stroke="currentColor" strokeWidth="2"
                                    strokeLinecap="round" strokeLinejoin="round"
                                    opacity="0.75"/>
                                <path d="M6 16 L10.5 16 L12.5 9 L16 23 L19.5 11 L21.5 16 L26 16"
                                    stroke="#3b9eff" strokeWidth="2"
                                    strokeLinecap="round" strokeLinejoin="round"/>
                            </svg>
                        </div>
                        <div className='BrandText'>
                            <h1 className='Wordmark'>TRACE</h1>
                            <div className='Expansion'>
                                <AcronymWord word='Temporal' highlightIndex={0} />
                                {' '}
                                <AcronymWord word='Recognition' highlightIndex={0} />
                                <span className='AOf'> of </span>
                                <AcronymWord word='Animal' highlightIndex={0} />
                                {' '}
                                <span className='AOf'>Behaviors </span>
                                <AcronymWord word='Captured' highlightIndex={0} />
                                <span className='AOf'> from </span>
                                <AcronymWord word='Video' highlightIndex={3} />
                            </div>
                        </div>
                    </div>
                    <p className='Tagline'>
                        Temporal action detection for animal behavior analysis.
                    </p>
                </header>

                {/* Bottom-left: Navigation */}
                <aside className='SideNav'>
                    <nav className='TabNav' aria-label='Workspace navigation'>
                        <div className='TabNavLabel'>Workspace</div>
                        <div className='TabGroup' role='tablist'>
                            <button
                                className={`TabBtn ${mode === 'annotate' ? 'active' : ''}`}
                                onClick={() => setMode('annotate')}
                                type='button'
                                role='tab'
                                aria-selected={mode === 'annotate'}
                                aria-label='Annotate tab'
                            >
                                <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
                                    <rect x="2" y="2" width="14" height="14" rx="2" stroke="currentColor" strokeWidth="1.3"/>
                                    <path d="M5 9h8M5 6.5h5M5 11.5h6" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round"/>
                                </svg>
                                <span>Annotate</span>
                            </button>
                            <button
                                className={`TabBtn ${mode === 'pipeline' ? 'active' : ''}`}
                                onClick={() => setMode('pipeline')}
                                type='button'
                                role='tab'
                                aria-selected={mode === 'pipeline'}
                                aria-label='Run Model tab'
                            >
                                <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
                                    <path d="M3 13L6 7l3 4 3-7 3 5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
                                </svg>
                                <span>Run Model</span>
                            </button>
                            <button
                                className={`TabBtn ${mode === 'tutorial' ? 'active' : ''}`}
                                onClick={() => setMode('tutorial')}
                                type='button'
                                role='tab'
                                aria-selected={mode === 'tutorial'}
                                aria-label='Tutorial tab'
                            >
                                <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
                                    <path d="M2 4.5C2 3.67 2.67 3 3.5 3h4L9 4.5H14.5c.83 0 1.5.67 1.5 1.5v7.5c0 .83-.67 1.5-1.5 1.5h-11C2.67 15 2 14.33 2 13.5V4.5z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round"/>
                                    <path d="M7 9.5l2.5 1.5L7 12.5v-3z" fill="currentColor" opacity="0.7"/>
                                </svg>
                                <span>Tutorial</span>
                            </button>
                        </div>

                    </nav>
                </aside>

                {/* Bottom-right: Workspace */}
                <section className='WorkspacePanel' key={mode}>
                    {mode === 'annotate' && <ImagesDropZone />}
                    {mode === 'pipeline' && <PipelineBuilder pipeline={pipeline} />}
                    {mode === 'tutorial' && <TutorialPanel />}
                </section>
            </div>
        </div>
    );
};

const mapStateToProps = (state: AppState) => ({
    homeTab: state.general.homeTab,
});

const mapDispatchToProps = {
    updateHomeTab,
};

export default connect(mapStateToProps, mapDispatchToProps)(MainView);
