import {ImageDataUtil} from "../ImageDataUtil";

describe('ImageDataUtil.createImageDataFromUrl', () => {
    it('omits frameRate when no value is supplied (legacy callers)', () => {
        const data = ImageDataUtil.createImageDataFromUrl('http://x/a.mp4', 'a.mp4');
        expect(data.videoUrl).toEqual('http://x/a.mp4');
        // The optional field must not be set so consumers pick up their
        // own `?? 30` defaults — Phase 4 of pts-based-frame-mapping.md
        // contract.
        expect(data.frameRate).toBeUndefined();
    });

    it('records a positive frameRate from the backend', () => {
        const data = ImageDataUtil.createImageDataFromUrl(
            'http://x/cfr.mp4', 'cfr.mp4', 29.97,
        );
        expect(data.frameRate).toEqual(29.97);
    });

    it('ignores a non-positive frameRate (treats it as missing)', () => {
        // ffprobe occasionally returns 0 or NaN for malformed
        // containers; the helper must not poison ImageData with that.
        const data = ImageDataUtil.createImageDataFromUrl(
            'http://x/bad.mp4', 'bad.mp4', 0,
        );
        expect(data.frameRate).toBeUndefined();
    });
});
