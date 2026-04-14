export class TimeUtil {
    /**
     * 格式化时间为 HH:MM:SS.sss 格式，并计算帧数
     * @param time 时间（秒）
     * @param frameRate 帧率
     * @returns 格式化的时间和帧数
     */
    public static formatTimeWithFrame(time: number, frameRate: number = 30): { formattedTime: string, frame: number } {
        const hours = Math.floor(time / 3600);
        const minutes = Math.floor((time % 3600) / 60);
        const seconds = Math.floor(time % 60);
        const milliseconds = Math.floor((time % 1) * 1000);
        const frame = Math.floor(time * frameRate);
        
        const formattedTime = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}.${milliseconds.toString().padStart(3, '0')}`;
        return { formattedTime, frame };
    }

    /**
     * 格式化时间为 HH:MM:SS.sss 格式
     * @param time 时间（秒）
     * @returns 格式化的时间字符串
     */
    public static formatTime(time: number): string {
        const { formattedTime } = TimeUtil.formatTimeWithFrame(time);
        return formattedTime;
    }

    /**
     * 从时间戳字符串解析为秒数
     * @param timestamp 时间戳字符串 (HH:MM:SS 或 HH:MM:SS.sss 或 xxx.xxxs)
     * @returns 秒数
     */
    public static parseTimestamp(timestamp: string): number {
        // Handle simple second format like "1.500s"
        if (timestamp.endsWith('s') && !timestamp.includes(':')) {
            return parseFloat(timestamp.slice(0, -1));
        }
        
        // Handle HH:MM:SS format
        const parts = timestamp.split(':');
        if (parts.length !== 3) return 0;

        const [hours, minutes, seconds] = parts;
        const secondsParts = seconds.split('.');
        const wholeSeconds = parseInt(secondsParts[0]);
        const milliseconds = secondsParts.length > 1 ? parseInt(secondsParts[1]) / 1000 : 0;

        return parseInt(hours) * 3600 + parseInt(minutes) * 60 + wholeSeconds + milliseconds;
    }

    /**
     * 计算给定时间对应的帧数
     * @param time 时间（秒）
     * @param frameRate 帧率
     * @returns 帧数
     */
    public static calculateFrame(time: number, frameRate: number = 30): number {
        return Math.floor(time * frameRate);
    }
} 