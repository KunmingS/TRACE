import { v4 as uuidv4 } from 'uuid';
import { VideoData, VideoFrame } from '../store/labels/types';
import { ImageDataUtil } from './ImageDataUtil';

export class VideoUtil {
    public static createVideoDataFromFile(fileData: File): VideoData {
        return {
            id: uuidv4(),
            fileData,
            loadStatus: false,
            duration: 0,
            fps: 0,
            width: 0,
            height: 0,
            frames: [],
            currentFrameIndex: 0
        };
    }

    public static async extractFrames(videoFile: File): Promise<VideoFrame[]> {
        return new Promise((resolve, reject) => {
            const video = document.createElement('video');
            video.src = URL.createObjectURL(videoFile);

            video.onloadedmetadata = () => {
                const duration = video.duration;
                const fps = 30;
                const width = video.videoWidth;
                const height = video.videoHeight;

                const frames: VideoFrame[] = [];
                const frameCount = Math.floor(duration * fps);

                for (let i = 0; i < frameCount; i++) {
                    const timestamp = i / fps;
                    video.currentTime = timestamp;

                    const canvas = document.createElement('canvas');
                    canvas.width = width;
                    canvas.height = height;
                    const ctx = canvas.getContext('2d');

                    if (ctx) {
                        ctx.drawImage(video, 0, 0, width, height);
                        const imageData = ImageDataUtil.createImageDataFromFileData(
                            new File([canvas.toDataURL()], `frame_${i}.jpg`, { type: 'image/jpeg' })
                        );

                        frames.push({
                            id: uuidv4(),
                            frameIndex: i,
                            timestamp,
                            imageData: {
                                ...imageData,
                                frameIndex: i,
                                timestamp
                            },
                            isKeyFrame: i % fps === 0
                        });
                    }
                }

                resolve(frames);
            };

            video.onerror = () => {
                reject(new Error('Failed to load video'));
            };
        });
    }
}
