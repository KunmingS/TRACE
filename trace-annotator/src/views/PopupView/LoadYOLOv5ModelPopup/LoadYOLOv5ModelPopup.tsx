import React from 'react';
import './LoadYOLOv5ModelPopup.scss'
import {GenericYesNoPopup} from '../GenericYesNoPopup/GenericYesNoPopup';
import {PopupActions} from '../../../logic/actions/PopupActions';
import {AppState} from '../../../store';
import {connect} from 'react-redux';
import {PopupWindowType} from '../../../data/enums/PopupWindowType';
import {GeneralActionTypes} from '../../../store/general/types';
import {updateActivePopupType} from '../../../store/general/actionCreators';
import {submitNewNotification} from '../../../store/notifications/actionCreators';
import {INotification, NotificationsActionType} from '../../../store/notifications/types';

interface IProps {
    updateActivePopupTypeAction: (activePopupType: PopupWindowType) => GeneralActionTypes;
    submitNewNotificationAction: (notification: INotification) => NotificationsActionType;
}

const LoadYOLOv5ModelPopup: React.FC<IProps> = ({ updateActivePopupTypeAction }) => {
    const onReject = () => {
        PopupActions.close();
    }

    const renderContent = () => {
        return (<div className='load-yolo-v5-model-popup'>
            <div className='message'>
                YOLOv5 model loading has been removed in this version.
            </div>
        </div>);
    }

    return (
        <GenericYesNoPopup
            title={'Load YOLOv5 model'}
            renderContent={renderContent}
            disableAcceptButton={true}
            acceptLabel={'Use model!'}
            onAccept={() => {}}
            rejectLabel={'Back'}
            onReject={onReject}
        />
    );
}

const mapDispatchToProps = {
    updateActivePopupTypeAction: updateActivePopupType,
    submitNewNotificationAction: submitNewNotification
};

const mapStateToProps = (state: AppState) => ({});

export default connect(
    mapStateToProps,
    mapDispatchToProps
)(LoadYOLOv5ModelPopup);
