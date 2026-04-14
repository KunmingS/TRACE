import React from 'react';
import './App.scss';
import MainView from './views/MainView/MainView';
import {AppState} from './store';
import {connect} from 'react-redux';
import PopupView from './views/PopupView/PopupView';
import MobileMainView from './views/MobileMainView/MobileMainView';
import {Settings} from './settings/Settings';
import {SizeItUpView} from './views/SizeItUpView/SizeItUpView';
import {PlatformModel} from './staticModels/PlatformModel';
import classNames from 'classnames';
import NotificationsView from './views/NotificationsView/NotificationsView';
import { GeneralSelector } from './store/selectors/GeneralSelector';
import EditorView from './views/EditorView/EditorView';

interface IProps {
    projectType: string;
    windowSize: {
        width: number;
        height: number;
    };
}

const App: React.FC<IProps> = ({ projectType, windowSize }) => {
    const selectRoute = () => {
        if (!!PlatformModel.mobileDeviceData.manufacturer && !!PlatformModel.mobileDeviceData.os)
            return <MobileMainView/>;
        if (!projectType) {
            return <MainView/>;
        }
        if (windowSize.height < Settings.EDITOR_MIN_HEIGHT || windowSize.width < Settings.EDITOR_MIN_WIDTH) {
            return <SizeItUpView/>;
        }
        return <EditorView/>;
    };

    return (
        <div className={classNames('App')} draggable={false}>
            {selectRoute()}
            <PopupView/>
            <NotificationsView/>
        </div>
    );
};

const mapStateToProps = (state: AppState) => ({
    projectType: GeneralSelector.getProjectType(),
    windowSize: GeneralSelector.getWindowSize(),
});

export default connect(
    mapStateToProps
)(App);
