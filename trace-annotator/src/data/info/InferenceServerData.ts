import {InferenceServerType} from '../enums/InferenceServerType';

export type InferenceServerData = {
    name: string;
    imageSrc: string;
    imageAlt: string;
    isDisabled: boolean;
}

export const InferenceServerDataMap: Record<InferenceServerType, InferenceServerData> = {
    [InferenceServerType.ROBOFLOW]: {
        name: 'Roboflow',
        imageSrc: 'ico/roboflow-logo.png',
        imageAlt: 'roboflow',
        isDisabled: false
    }
};
