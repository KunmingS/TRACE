import React, {useEffect} from 'react';
import './EditorView.scss';
import EditorContainer from './EditorContainer/EditorContainer';
import {PopupWindowType} from '../../data/enums/PopupWindowType';
import {AppState} from '../../store';
import {connect} from 'react-redux';
import classNames from 'classnames';
import TopNavigationBar from './TopNavigationBar/TopNavigationBar';
import {updateHomeTab, updateProjectData} from '../../store/general/actionCreators';
import {
    updateActiveImageIndex,
    updateActiveLabelNameId,
    updateFirstLabelCreatedFlag,
    updateImageData,
    updateLabelNames
} from '../../store/labels/actionCreators';
import {ProjectData} from '../../store/general/types';
import {ImageData, LabelName} from '../../store/labels/types';

interface IProps {
    activePopupType: PopupWindowType;
    updateProjectDataAction: (projectData: ProjectData) => any;
    updateActiveImageIndex: (index: number) => any;
    updateActiveLabelNameId: (id: string) => any;
    updateLabelNames: (labels: LabelName[]) => any;
    updateImageData: (imageData: ImageData[]) => any;
    updateFirstLabelCreatedFlag: (flag: boolean) => any;
    updateHomeTab: typeof updateHomeTab;
}

const EditorView: React.FC<IProps> = (props) => {
    const {activePopupType} = props;

    // The app uses Redux-driven view switching (App.tsx picks MainView vs
    // EditorView off `projectType`), not URL routing, so the browser never
    // accumulated history for the home → editor transition. Pressing the
    // back button used to leave the app entirely. Push a sentinel entry on
    // mount and treat the resulting popstate as a request to return home —
    // same effect as clicking the Home button in TopNavigationBar.
    useEffect(() => {
        window.history.pushState({traceEditor: true}, '');

        const goHome = () => {
            props.updateHomeTab('annotate');
            props.updateActiveLabelNameId(null);
            props.updateLabelNames([]);
            props.updateProjectDataAction({type: null, name: 'my-project-name'});
            props.updateActiveImageIndex(null);
            props.updateImageData([]);
            props.updateFirstLabelCreatedFlag(false);
        };

        window.addEventListener('popstate', goHome);
        return () => window.removeEventListener('popstate', goHome);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const getClassName = () => {
        return classNames(
            'EditorView',
            {
                'withPopup': !!activePopupType
            }
        );
    };

    return (
        <div
            className={getClassName()}
            draggable={false}
        >
            <TopNavigationBar/>
            <EditorContainer/>
        </div>
    );
};

const mapStateToProps = (state: AppState) => ({
    activePopupType: state.general.activePopupType
});

const mapDispatchToProps = {
    updateProjectDataAction: updateProjectData,
    updateActiveImageIndex,
    updateActiveLabelNameId,
    updateLabelNames,
    updateImageData,
    updateFirstLabelCreatedFlag,
    updateHomeTab,
};

export default connect(
    mapStateToProps,
    mapDispatchToProps
)(EditorView);
