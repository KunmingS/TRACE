import { v4 as uuidv4 } from 'uuid';
import { VideoData } from '../store/labels/types';

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
}
