import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import PathPicker from './PathPicker';
import { PlatformProvider } from './PlatformContext';
import { PlatformInfo } from './pathAdapter';

jest.mock('../../../config', () => ({ API_URL: '' }));

const jsonResponse = (body: any) => Promise.resolve({
    ok: true,
    json: async () => body,
});

const POSIX_FIXTURE: PlatformInfo = {
    os: 'posix',
    sep: '/',
    home: '/home/user',
    roots: [{ path: '/', label: '/', volumeName: '', free: null, total: null }],
};

const WINDOWS_FIXTURE: PlatformInfo = {
    os: 'windows',
    sep: '\\',
    home: 'C:\\Users\\skm',
    roots: [
        { path: 'C:\\', label: 'C:', volumeName: 'Local Disk', free: 53_687_091_200, total: 256_000_000_000 },
        { path: 'D:\\', label: 'D:', volumeName: 'Data', free: 1_000_000_000_000, total: 2_000_000_000_000 },
    ],
};

const StatefulPicker = (
    props: Partial<React.ComponentProps<typeof PathPicker>> & { override?: PlatformInfo },
) => {
    const [value, setValue] = React.useState(props.value || '');
    return (
        <PlatformProvider override={props.override || POSIX_FIXTURE}>
            <PathPicker
                value={value}
                onChange={setValue}
                onSubmit={props.onSubmit}
                placeholder={props.placeholder || '/path'}
                mode={props.mode}
                extensions={props.extensions}
                disabled={props.disabled}
                storageKey={props.storageKey}
                previewExtensions={props.previewExtensions}
            />
        </PlatformProvider>
    );
};

describe('PathPicker (POSIX)', () => {
    beforeEach(() => {
        jest.useFakeTimers();
        global.fetch = jest.fn() as any;
        localStorage.clear();
    });

    afterEach(() => {
        jest.runOnlyPendingTimers();
        jest.useRealTimers();
        jest.resetAllMocks();
    });

    test('opens directory browser on focus and lists home dir contents', async () => {
        const fetchMock = global.fetch as jest.Mock;
        fetchMock.mockImplementationOnce(() => jsonResponse({
            entries: [
                { name: 'Documents', type: 'dir' },
                { name: 'Downloads', type: 'dir' },
            ],
            resolved: '/home/user',
        }));

        render(<StatefulPicker value='' placeholder='/path' />);

        const input = screen.getByPlaceholderText('/path');
        await act(async () => {
            fireEvent.focus(input);
            await Promise.resolve();
        });

        await waitFor(() => {
            expect(screen.getByText('Documents')).toBeInTheDocument();
            expect(screen.getByText('Downloads')).toBeInTheDocument();
        });
    });

    test('clicking a folder entry navigates into it', async () => {
        const fetchMock = global.fetch as jest.Mock;
        fetchMock
            .mockImplementationOnce(() => jsonResponse({
                entries: [
                    { name: 'projects', type: 'dir' },
                    { name: 'data', type: 'dir' },
                ],
                resolved: '/home/user',
            }))
            .mockImplementationOnce(() => jsonResponse({
                entries: [{ name: 'myapp', type: 'dir' }],
                resolved: '/home/user/projects',
            }));

        render(<StatefulPicker value='' placeholder='/path' />);

        const input = screen.getByPlaceholderText('/path');
        await act(async () => {
            fireEvent.focus(input);
            await Promise.resolve();
        });

        const projectsEntry = await screen.findByText('projects');
        await act(async () => {
            fireEvent.mouseDown(projectsEntry);
            await Promise.resolve();
        });

        await waitFor(() => {
            expect((input as HTMLInputElement).value).toBe('/home/user/projects/');
        });

        await waitFor(() => {
            expect(screen.getByText('myapp')).toBeInTheDocument();
        });
    });

    test('clicking .. navigates to parent directory', async () => {
        const fetchMock = global.fetch as jest.Mock;
        fetchMock
            .mockImplementationOnce(() => jsonResponse({
                entries: [{ name: 'sub', type: 'dir' }],
                resolved: '/home/user/projects',
            }))
            .mockImplementationOnce(() => jsonResponse({
                entries: [
                    { name: 'projects', type: 'dir' },
                    { name: 'data', type: 'dir' },
                ],
                resolved: '/home/user',
            }));

        render(<StatefulPicker value='/home/user/projects/' placeholder='/path' />);

        const input = screen.getByPlaceholderText('/path');
        await act(async () => {
            fireEvent.focus(input);
            await Promise.resolve();
        });

        const parentEntry = await screen.findByText('..');
        await act(async () => {
            fireEvent.mouseDown(parentEntry);
            await Promise.resolve();
        });

        await waitFor(() => {
            expect((input as HTMLInputElement).value).toBe('/home/user/');
        });
    });

    test('typing filters the directory listing', async () => {
        const fetchMock = global.fetch as jest.Mock;
        fetchMock.mockImplementationOnce(() => jsonResponse({
            entries: [
                { name: 'Documents', type: 'dir' },
                { name: 'Downloads', type: 'dir' },
                { name: 'Desktop', type: 'dir' },
            ],
            resolved: '/home/user',
        }));

        render(<StatefulPicker value='' placeholder='/path' />);

        const input = screen.getByPlaceholderText('/path');
        await act(async () => {
            fireEvent.focus(input);
            await Promise.resolve();
        });

        await waitFor(() => {
            expect(screen.getByText('Documents')).toBeInTheDocument();
        });

        await act(async () => {
            fireEvent.change(input, { target: { value: '/home/user/Do' } });
        });

        const entryByName = (name: string) =>
            screen.queryByText((_, el) =>
                el?.tagName === 'SPAN'
                && el.classList.contains('EntryName')
                && el.textContent === name
            );

        await waitFor(() => {
            expect(entryByName('Documents')).toBeInTheDocument();
            expect(entryByName('Downloads')).toBeInTheDocument();
            expect(entryByName('Desktop')).not.toBeInTheDocument();
        });
    });

    test('shows breadcrumbs when navigating', async () => {
        const fetchMock = global.fetch as jest.Mock;
        fetchMock.mockImplementationOnce(() => jsonResponse({
            entries: [],
            resolved: '/media/data/clips',
        }));

        render(<StatefulPicker value='/media/data/clips/' placeholder='/path' />);

        const input = screen.getByPlaceholderText('/path');
        await act(async () => {
            fireEvent.focus(input);
            await Promise.resolve();
        });

        await waitFor(() => {
            expect(screen.getByText('media')).toBeInTheDocument();
            expect(screen.getByText('data')).toBeInTheDocument();
            expect(screen.getByText('clips')).toBeInTheDocument();
        });
    });

    test('shows files in file mode with extensions', async () => {
        const fetchMock = global.fetch as jest.Mock;
        fetchMock.mockImplementationOnce(() => jsonResponse({
            entries: [
                { name: 'subdir', type: 'dir' },
                { name: 'video.mp4', type: 'file' },
            ],
            resolved: '/data',
        }));

        render(
            <StatefulPicker
                value='/data/'
                placeholder='/video'
                mode='file'
                extensions='.mp4,.avi'
            />
        );

        const input = screen.getByPlaceholderText('/video');
        await act(async () => {
            fireEvent.focus(input);
            await Promise.resolve();
        });

        await waitFor(() => {
            expect(screen.getByText('subdir')).toBeInTheDocument();
            expect(screen.getByText('video.mp4')).toBeInTheDocument();
        });

        const lsCall = fetchMock.mock.calls.find(
            (c: any) => typeof c[0] === 'string' && c[0].includes('/api/ls')
        );
        expect(lsCall).toBeTruthy();
        expect(lsCall[0]).toContain('extensions=');
    });

    test('groups preview files into related recording sets', async () => {
        const fetchMock = global.fetch as jest.Mock;
        fetchMock.mockImplementationOnce(() => jsonResponse({
            entries: [
                { name: 'snap', type: 'dir' },
                { name: 'movie.mkv', type: 'file' },
                { name: 'movie.mkv.remux.mp4', type: 'file' },
                { name: 'movie.csv', type: 'file' },
                { name: 'movie_raterA.csv', type: 'file' },
                { name: 'movie10.csv', type: 'file' },
            ],
            resolved: '/data',
        }));

        render(
            <StatefulPicker
                value='/data/'
                placeholder='/path'
                previewExtensions='.mp4,.avi,.mov,.mkv,.webm,.csv'
            />
        );

        const input = screen.getByPlaceholderText('/path');
        await act(async () => {
            fireEvent.focus(input);
            await Promise.resolve();
        });

        await waitFor(() => {
            expect(screen.getByText('Related Sets')).toBeInTheDocument();
            expect(screen.getByText('movie.mkv')).toBeInTheDocument();
            expect(screen.getByText('movie.mkv.remux.mp4')).toBeInTheDocument();
            expect(screen.getByText('movie.csv')).toBeInTheDocument();
            expect(screen.getByText('movie_raterA.csv')).toBeInTheDocument();
        });

        expect(screen.getByText('2 csv')).toBeInTheDocument();
        expect(screen.getByText('movie10.csv')).toBeInTheDocument();
    });
});

describe('PathPicker (Windows)', () => {
    beforeEach(() => {
        jest.useFakeTimers();
        global.fetch = jest.fn() as any;
        localStorage.clear();
    });

    afterEach(() => {
        jest.runOnlyPendingTimers();
        jest.useRealTimers();
        jest.resetAllMocks();
    });

    test('focus on empty value lands in home dir, not drive list', async () => {
        const fetchMock = global.fetch as jest.Mock;
        fetchMock.mockImplementationOnce(() => jsonResponse({
            entries: [{ name: 'videos', type: 'dir' }],
            resolved: 'C:\\Users\\skm',
        }));

        render(<StatefulPicker value='' placeholder='winpath' override={WINDOWS_FIXTURE} />);

        const input = screen.getByPlaceholderText('winpath');
        await act(async () => {
            fireEvent.focus(input);
            await Promise.resolve();
        });

        await waitFor(() => {
            // Value should be the home dir with trailing backslash (no forward slash).
            expect((input as HTMLInputElement).value).toBe('C:\\Users\\skm\\');
            expect(screen.getByText('videos')).toBeInTheDocument();
        });
    });

    test('navigating into a Windows folder builds backslash-separated paths', async () => {
        const fetchMock = global.fetch as jest.Mock;
        fetchMock
            .mockImplementationOnce(() => jsonResponse({
                entries: [{ name: 'videos', type: 'dir' }],
                resolved: 'C:\\Users\\skm',
            }))
            .mockImplementationOnce(() => jsonResponse({
                entries: [{ name: 'session-01', type: 'dir' }],
                resolved: 'C:\\Users\\skm\\videos',
            }));

        render(<StatefulPicker value='' placeholder='winpath' override={WINDOWS_FIXTURE} />);

        const input = screen.getByPlaceholderText('winpath');
        await act(async () => {
            fireEvent.focus(input);
            await Promise.resolve();
        });

        const videosEntry = await screen.findByText('videos');
        await act(async () => {
            fireEvent.mouseDown(videosEntry);
            await Promise.resolve();
        });

        await waitFor(() => {
            expect((input as HTMLInputElement).value).toBe('C:\\Users\\skm\\videos\\');
            expect(screen.getByText('session-01')).toBeInTheDocument();
        });
    });

    test('drive-list view renders enumerated drives and navigates into them', async () => {
        const fetchMock = global.fetch as jest.Mock;
        // First focus → empty path → backend would return drives; we mirror that.
        fetchMock
            .mockImplementationOnce(() => jsonResponse({
                entries: [{ name: 'Users', type: 'dir' }],
                resolved: 'C:\\Users\\skm',
            }));

        render(<StatefulPicker value='' placeholder='winpath' override={WINDOWS_FIXTURE} />);

        const input = screen.getByPlaceholderText('winpath');
        await act(async () => {
            fireEvent.focus(input);
            await Promise.resolve();
        });

        // Now jump to drive-list view by clicking "This PC" in the breadcrumb.
        // Setup a second fetch for the drive-list response.
        fetchMock.mockImplementationOnce(() => jsonResponse({
            entries: [],
            resolved: '',
            roots: WINDOWS_FIXTURE.roots,
        }));
        // Then a third fetch for navigating into D:\.
        fetchMock.mockImplementationOnce(() => jsonResponse({
            entries: [{ name: 'datasets', type: 'dir' }],
            resolved: 'D:\\',
        }));

        const thisPcBtn = await screen.findByRole('button', { name: /This PC/i });
        await act(async () => {
            fireEvent.click(thisPcBtn);
            await Promise.resolve();
        });

        // Both drives should be listed by label.
        await waitFor(() => {
            const driveLabels = screen.getAllByText((_, el) =>
                el?.classList.contains('EntryName')
                && (el.textContent === 'C:' || el.textContent === 'D:')
            );
            expect(driveLabels.length).toBeGreaterThanOrEqual(2);
        });

        const dDrive = screen.getAllByText((_, el) =>
            el?.classList.contains('EntryName') && el.textContent === 'D:'
        )[0];
        await act(async () => {
            fireEvent.mouseDown(dDrive);
            await Promise.resolve();
        });

        await waitFor(() => {
            expect((input as HTMLInputElement).value).toBe('D:\\');
            expect(screen.getByText('datasets')).toBeInTheDocument();
        });
    });

    test('pasting a path with forward slashes normalizes to backslashes', async () => {
        const fetchMock = global.fetch as jest.Mock;
        fetchMock
            .mockImplementationOnce(() => jsonResponse({
                entries: [],
                resolved: 'C:\\Users\\skm',
            }))
            .mockImplementationOnce(() => jsonResponse({
                entries: [{ name: 'clip.mp4', type: 'file' }],
                resolved: 'C:\\Users\\skm\\videos',
            }));

        render(<StatefulPicker value='' placeholder='winpath' override={WINDOWS_FIXTURE} />);

        const input = screen.getByPlaceholderText('winpath');
        await act(async () => {
            fireEvent.focus(input);
            await Promise.resolve();
        });

        // Simulate pasting a forward-slash path.
        await act(async () => {
            fireEvent.change(input, { target: { value: 'C:/Users/skm/videos/' } });
            jest.advanceTimersByTime(200);
            await Promise.resolve();
        });

        await waitFor(() => {
            expect((input as HTMLInputElement).value).toBe('C:\\Users\\skm\\videos\\');
        });
    });
});
