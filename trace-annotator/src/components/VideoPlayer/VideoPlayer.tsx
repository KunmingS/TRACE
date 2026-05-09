import React, { useEffect, useMemo, useRef, useState } from 'react';
import { VideoData, VideoFrame } from '../../store/labels/types';
import './VideoPlayer.scss';

interface Props {
    videoData: VideoData;
    currentFrame: VideoFrame;
    onFrameChange: (frameIndex: number) => void;
    onPlay: () => void;
    onPause: () => void;
    isPlaying: boolean;
}

export const VideoPlayer: React.FC<Props> = ({
    videoData,
    currentFrame,
    onFrameChange,
    onPlay,
    onPause,
    isPlaying
}) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const [currentTime, setCurrentTime] = useState(0);

    // Create the object URL once per file — recreating every render leaks memory
    // and forces the <video> element to reload on each parent re-render.
    const videoSrc = useMemo(
        () => URL.createObjectURL(videoData.fileData),
        [videoData.fileData]
    );
    useEffect(() => () => URL.revokeObjectURL(videoSrc), [videoSrc]);

    useEffect(() => {
        if (videoRef.current) {
            videoRef.current.currentTime = currentFrame.timestamp;
        }
    }, [currentFrame]);

    const handleTimeUpdate = () => {
        if (videoRef.current) {
            const time = videoRef.current.currentTime;
            setCurrentTime(time);
            const frameIndex = Math.floor(time * videoData.fps);
            onFrameChange(frameIndex);
        }
    };

    const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
        const time = parseFloat(e.target.value);
        if (videoRef.current) {
            videoRef.current.currentTime = time;
            setCurrentTime(time);
        }
    };

    const handlePlayPause = () => {
        if (videoRef.current) {
            if (isPlaying) {
                videoRef.current.pause();
                onPause();
            } else {
                videoRef.current.play();
                onPlay();
            }
        }
    };

    const handleFrameStep = (direction: 'forward' | 'backward') => {
        const newFrameIndex = direction === 'forward' 
            ? Math.min(currentFrame.frameIndex + 1, videoData.frames.length - 1)
            : Math.max(currentFrame.frameIndex - 1, 0);
        onFrameChange(newFrameIndex);
    };

    return (
        <div className="VideoPlayer">
            <video
                ref={videoRef}
                src={videoSrc}
                onTimeUpdate={handleTimeUpdate}
                className="video-element"
            />
            
            <div className="video-controls">
                <button type='button' onClick={() => handleFrameStep('backward')} aria-label='Step backward'>
                    <i className="fas fa-step-backward" />
                </button>

                <button type='button' onClick={handlePlayPause} aria-label={isPlaying ? 'Pause' : 'Play'}>
                    <i className={`fas fa-${isPlaying ? 'pause' : 'play'}`} />
                </button>

                <button type='button' onClick={() => handleFrameStep('forward')} aria-label='Step forward'>
                    <i className="fas fa-step-forward" />
                </button>
                
                <input
                    type="range"
                    min={0}
                    max={videoData.duration}
                    step={1/videoData.fps}
                    value={currentTime}
                    onChange={handleSeek}
                    className="timeline"
                />
                
                <div className="time-display">
                    {formatTime(currentTime)} / {formatTime(videoData.duration)}
                </div>
            </div>
        </div>
    );
};

const formatTime = (seconds: number): string => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}; 