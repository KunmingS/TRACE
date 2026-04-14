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
    projectData: state.general.projectData
});

export default connect(
    mapStateToProps,
    mapDispatchToProps
)(TopNavigationBar);
