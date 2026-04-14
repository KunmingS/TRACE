import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import PathPicker from './PathPicker';

jest.mock('../../../config', () => ({ API_URL: '' }));

const jsonResponse = (body: any) => Promise.resolve({
    ok: true,
    json: async () => body,
});

const StatefulPicker = (props: Partial<React.ComponentProps<typeof PathPicker>>) => {
    const [value, setValue] = React.useState(props.value || '');
    return (
        <PathPicker
            value={value}
            onChange={setValue}
            onSubmit={props.onSubmit}
            placeholder={props.placeholder || '/path'}
            mode={props.mode}
            extensions={props.extensions}
            disabled={props.disabled}
            storageKey={props.storageKey}
        />
    );
};

describe('PathPicker', () => {
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
        fetchMock
            .mockImplementationOnce(() => jsonResponse({ home: '/home/user' }))
            .mockImplementationOnce(() => jsonResponse({
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
            .mockImplementationOnce(() => jsonResponse({ home: '/home/user' }))
            .mockImplementationOnce(() => jsonResponse({
                entries: [
                    { name: 'projects', type: 'dir' },
                    { name: 'data', type: 'dir' },
                ],
                resolved: '/home/user',
            }))
            .mockImplementationOnce(() => jsonResponse({
                entries: [
                    { name: 'myapp', type: 'dir' },
                ],
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
            .mockImplementationOnce(() => jsonResponse({ home: '/home/user' }))
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
        fetchMock
            .mockImplementationOnce(() => jsonResponse({ home: '/home/user' }))
            .mockImplementationOnce(() => jsonResponse({
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

        // Now type to filter — change value to partial name
        await act(async () => {
            fireEvent.change(input, { target: { value: '/home/user/Do' } });
        });

        // Should filter to entries starting with "Do"
        await waitFor(() => {
            expect(screen.getByText('Documents')).toBeInTheDocument();
            expect(screen.getByText('Downloads')).toBeInTheDocument();
            expect(screen.queryByText('Desktop')).not.toBeInTheDocument();
        });
    });

    test('shows breadcrumbs when navigating', async () => {
        const fetchMock = global.fetch as jest.Mock;
        fetchMock
            .mockImplementation(() => jsonResponse({ home: '/home/user' }))
            .mockImplementationOnce(() => jsonResponse({ home: '/home/user' }))
            .mockImplementationOnce(() => jsonResponse({
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
        fetchMock
            .mockImplementationOnce(() => jsonResponse({ home: '/home/user' }))
            .mockImplementationOnce(() => jsonResponse({
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

        // Verify /api/ls was called with extensions
        const lsCall = fetchMock.mock.calls.find(
            (c: any) => typeof c[0] === 'string' && c[0].includes('/api/ls')
        );
        expect(lsCall).toBeTruthy();
        expect(lsCall[0]).toContain('extensions=');
    });
});
