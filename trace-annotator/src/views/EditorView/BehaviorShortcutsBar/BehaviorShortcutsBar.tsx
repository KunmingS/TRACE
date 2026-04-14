import React from 'react';
import { connect } from 'react-redux';
import { AppState } from '../../../store';
import { ImageData, LabelName } from '../../../store/labels/types';
import { updateImageDataById } from '../../../store/labels/actionCreators';
import { toggleBehaviorClip } from '../../../utils/BehaviorUtil';
import './BehaviorShortcutsBar.scss';

interface IProps {
    imageData: ImageData;
    labelNames: LabelName[];
    currentTime: number;
    frameRate: number;
    updateImageDataById: (id: string, newImageData: ImageData) => any;
}

const BehaviorShortcutsBar: React.FC<IProps> = ({
    imageData,
    labelNames,
    currentTime,
    frameRate,
    updateImageDataById
}) => {
    const handleBadgeClick = (labelName: LabelName) => {
        if (!imageData) return;
        const updatedData = toggleBehaviorClip(labelName, imageData, currentTime, frameRate);
        updateImageDataById(imageData.id, updatedData);
    };

    const isRecording = (labelName: LabelName): boolean => {
        return imageData?.labelRects?.some(
            rect => rect.labelId === labelName.id && !rect.endTimestamp
        ) || false;
    };

    return (
        <div className="BehaviorShortcutsBar">
            <span className="BarLabel">Behaviors</span>
            {labelNames.map(ln => (
                <div
                    key={ln.id}
                    className={`ShortcutBadge ${isRecording(ln) ? 'recording' : ''}`}
                    onClick={() => handleBadgeClick(ln)}
                    title={`Press ${ln.shortcut || '?'} to toggle`}
                >
                    <div className="ColorDot" style={{ backgroundColor: ln.color || '#5297FF' }} />
                    {ln.shortcut && <span className="KeyBadge">{ln.shortcut}</span>}
                    <span className="BehaviorName">{ln.name}</span>
                    {isRecording(ln) && <span className="RecDot" />}
                </div>
            ))}
        </div>
    );
};

const mapStateToProps = (state: AppState) => ({
    imageData: state.labels.imagesData[state.labels.activeImageIndex],
    labelNames: state.labels.labels
});

const mapDispatchToProps = { updateImageDataById };

export default connect(mapStateToProps, mapDispatchToProps)(BehaviorShortcutsBar);
