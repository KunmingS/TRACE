import {Notification} from '../enums/Notification';

export type NotificationContent = {
    header: string;
    description: string;
}

export const NotificationsDataMap: Record<Notification, NotificationContent> = {
    [Notification.EMPTY_LABEL_NAME_ERROR]: {
        header: 'Empty label name',
        description: 'One of the label names is empty. Please fill all label name fields or remove unnecessary ones.'
    },
    [Notification.NON_UNIQUE_LABEL_NAMES_ERROR]: {
        header: 'Non-unique label names',
        description: 'Label names must be unique. Please correct duplicated label names.'
    },
    [Notification.UNSUPPORTED_INFERENCE_SERVER_MESSAGE]: {
        header: 'Unsupported inference server',
        description: 'This inference server type is not yet supported. Please choose a different option.'
    },
    [Notification.ROBOFLOW_INFERENCE_SERVER_ERROR]: {
        header: 'Roboflow connection error',
        description: 'Failed to connect to Roboflow inference server. Please check your model name and API key.'
    },
    [Notification.ANNOTATION_FILE_PARSE_ERROR]: {
        header: 'Annotation file parse error',
        description: 'Failed to parse the annotation file. Please check the file format and try again.'
    },
    [Notification.ANNOTATION_IMPORT_ASSERTION_ERROR]: {
        header: 'Annotation import error',
        description: 'An error occurred during annotation import. Please verify the annotation data and try again.'
    }
};
