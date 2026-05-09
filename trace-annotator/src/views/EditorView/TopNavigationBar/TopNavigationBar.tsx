import React from 'react';
import './TopNavigationBar.scss';
import StateBar from '../StateBar/StateBar';
import {AppState} from '../../../store';
import {connect} from 'react-redux';
import {updateProjectData, updateHomeTab} from '../../../store/general/actionCreators';
import {
    updateActiveImageIndex,
    updateActiveLabelNameId,
    updateFirstLabelCreatedFlag,
    updateImageData,
    updateLabelNames
} from '../../../store/labels/actionCreators';
import {ProjectData} from '../../../store/general/types';
import {ImageData, LabelName} from '../../../store/labels/types';

interface IProps {
    updateProjectDataAction: (projectData: ProjectData) => any;
    updateActiveImageIndex: (index: number) => any;
    updateActiveLabelNameId: (id: string) => any;
    updateLabelNames: (labels: LabelName[]) => any;
    updateImageData: (imageData: ImageData[]) => any;
    updateFirstLabelCreatedFlag: (flag: boolean) => any;
    updateHomeTab: typeof updateHomeTab;
    projectData: ProjectData;
    // The video file currently loaded into the editor and the CSV the user
    // picked from the per-video CSV list (`{base}.csv` plus `{base}_*.csv`
    // variants — rater A vs rater B, draft vs final). Both are shown in
    // the top-bar manifest so the active labeling is visible at a glance
    // even when the FileBrowser sidebar is collapsed.
    activeVideoName: string;
    activeCsvName: string;
}

const TopNavigationBar: React.FC<IProps> = (props) => {
    const exitEditor = () => {
        props.updateActiveLabelNameId(null);
        props.updateLabelNames([]);
        props.updateProjectDataAction({ type: null, name: 'my-project-name' });
        props.updateActiveImageIndex(null);
        props.updateImageData([]);
        props.updateFirstLabelCreatedFlag(false);
    };

    const goHome = () => {
        props.updateHomeTab('annotate');
        exitEditor();
    };

    const goToPipeline = () => {
        props.updateHomeTab('pipeline');
        exitEditor();
    };

    const hasActive = !!props.activeVideoName;

    return (
        <div className='TopNavigationBar'>
            <StateBar/>
            <div className='TopNavigationBarWrapper'>
                <div className='NavigationBarGroupWrapper'>
                    <button
                        className='HomeButton'
                        onClick={goHome}
                        type='button'
                        aria-label='Back to home'
                    >
                        <svg className='HomeButton__arrow' width="16" height="16" viewBox="0 0 16 16" fill="none">
                            <path d="M10 3L5 8l5 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                        <img
                            draggable={false}
                            alt='trace'
                            src='/trace-logo.svg'
                            className='HomeButton__logo'
                        />
                        <span className='HomeButton__label'>Home</span>
                    </button>
                    <span className='NavDivider' />
                    <button
                        className='PipelineButton'
                        onClick={goToPipeline}
                        type='button'
                        aria-label='Go to pipeline'
                    >
                        <svg width="14" height="14" viewBox="0 0 18 18" fill="none">
                            <path d="M3 13L6 7l3 4 3-7 3 5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                        <span>Pipeline</span>
                    </button>
                </div>

                {hasActive && (
                    <div
                        className='NowEditingPlate'
                        role='status'
                        aria-live='polite'
                        title={
                            props.activeCsvName
                                ? `Editing ${props.activeVideoName} → ${props.activeCsvName}`
                                : `Viewing ${props.activeVideoName} (no annotation file selected)`
                        }
                    >
                        <span className='NowEditingPlate__eyebrow'>Editing</span>
                        <span className='NowEditingPlate__row'>
                            <span className='NowEditingPlate__video' title={props.activeVideoName}>
                                <svg className='NowEditingPlate__icon video' width='12' height='12' viewBox='0 0 16 16' fill='none' aria-hidden='true'>
                                    <rect x='2.5' y='3.5' width='11' height='9' rx='1.2' stroke='currentColor' strokeWidth='1.1' fill='none' />
                                    <path d='M7 6.5 L10.2 8 L7 9.5 Z' fill='currentColor' />
                                </svg>
                                <span className='NowEditingPlate__videoName'>{props.activeVideoName}</span>
                            </span>
                            <svg className='NowEditingPlate__sep' width='14' height='10' viewBox='0 0 14 10' fill='none' aria-hidden='true'>
                                <path d='M1 5h11M9 1.5L12.5 5L9 8.5' stroke='currentColor' strokeWidth='1.2' strokeLinecap='round' strokeLinejoin='round' />
                            </svg>
                            {props.activeCsvName ? (
                                <span className='NowEditingPlate__csv' title={props.activeCsvName}>
                                    <svg className='NowEditingPlate__icon csv' width='12' height='12' viewBox='0 0 16 16' fill='none' aria-hidden='true'>
                                        <rect x='3' y='2' width='9' height='12' rx='1.2' stroke='currentColor' strokeWidth='1.1' fill='none' />
                                        <path d='M5 6h6M5 8.5h6M5 11h4' stroke='currentColor' strokeWidth='0.9' strokeLinecap='round' />
                                    </svg>
                                    <span className='NowEditingPlate__csvName'>{props.activeCsvName}</span>
                                </span>
                            ) : (
                                <span className='NowEditingPlate__csv NowEditingPlate__csv--empty'>
                                    <span className='NowEditingPlate__csvName'>no annotation file selected</span>
                                </span>
                            )}
                        </span>
                    </div>
                )}
            </div>
        </div>
    );
};

const mapDispatchToProps = {
    updateProjectDataAction: updateProjectData,
    updateActiveImageIndex,
    updateActiveLabelNameId,
    updateLabelNames,
    updateImageData,
    updateFirstLabelCreatedFlag,
    updateHomeTab,
};

const mapStateToProps = (state: AppState) => ({
    projectData: state.general.projectData,
    activeVideoName: state.labels.imagesData[state.labels.activeImageIndex]?.fileData?.name || '',
    activeCsvName: (state.labels.imagesData[state.labels.activeImageIndex] as any)?.csvName || '',
});

export default connect(
    mapStateToProps,
    mapDispatchToProps
)(TopNavigationBar);
