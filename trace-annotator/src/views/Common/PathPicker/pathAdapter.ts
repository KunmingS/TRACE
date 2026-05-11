// OS-aware path utilities. Built once from `/api/platform` and provided via
// PlatformContext. All the call sites that used to hard-code '/' as the
// separator route through here, so the only place that knows about Windows
// vs POSIX is this file and the drive-list renderer in PathPicker.
//
// Path conventions inside the picker:
//   * `currentDir` is the in-folder marker — '' means "drive list view"
//     (Windows only; POSIX never reaches it because '/' is always a folder).
//   * The user-facing `value` carries a trailing separator when the user has
//     committed to "I am inside this folder, list its contents", and does not
//     when the user is typing a partial name. handleChange() reads the
//     trailing sep to decide between fetchListing(dir) and a parent-fetch.

export interface DriveRoot {
    path: string;       // 'C:\\' or '/'
    label: string;      // 'C:' or '/'
    volumeName: string; // 'Local Disk' or '' (best-effort)
    free: number | null;
    total: number | null;
}

export interface PlatformInfo {
    os: 'windows' | 'posix';
    sep: string;
    home: string;
    roots: DriveRoot[];
}

export interface Crumb {
    segment: string;
    fullPath: string;
}

export interface PathAdapter extends PlatformInfo {
    isRoot(path: string): boolean;
    isAbsolute(path: string): boolean;
    /** Unify separators to the OS-native one and collapse duplicates. */
    normalize(path: string): string;
    join(dir: string, name: string): string;
    /** Parent of a path; returns '' to signal "go to drive list" on Windows. */
    parent(path: string): string;
    basename(path: string): string;
    splitCrumbs(path: string): Crumb[];
    /** Append separator if missing, except for empty strings. */
    ensureTrailing(path: string): string;
    /** Remove trailing separator(s) but leave drive root / POSIX root intact. */
    stripTrailing(path: string): string;
    /** Expand a leading '~' against `home`. Returns input unchanged otherwise. */
    resolveTilde(path: string): string;
    /** Placeholder shown in the input when empty. */
    samplePath: string;
}

function makeWindowsAdapter(info: PlatformInfo): PathAdapter {
    const normalize = (p: string): string => {
        if (!p) return p;
        // Tildes are expanded separately — never touch the lead '~'.
        if (p === '~' || p.startsWith('~\\') || p.startsWith('~/')) {
            return '~' + (p.length > 1 ? normalize(p.slice(1)) : '');
        }
        return p.replace(/[\\/]+/g, '\\');
    };

    const isAbsolute = (p: string): boolean =>
        /^[A-Za-z]:[\\/]/.test(p);

    const isRoot = (p: string): boolean => /^[A-Za-z]:\\?$/.test(p);

    const stripTrailing = (p: string): string => {
        if (!p) return p;
        if (isRoot(p)) {
            // 'C:' → 'C:\' ; 'C:\' stays as-is.
            return p.length === 2 ? p + '\\' : p;
        }
        return p.replace(/[\\/]+$/, '');
    };

    const ensureTrailing = (p: string): string => {
        if (!p) return p;
        return /[\\/]$/.test(p) ? p : p + '\\';
    };

    const join = (dir: string, name: string): string => {
        if (!dir) return name;
        if (isRoot(dir)) return ensureTrailing(dir) + name;
        return stripTrailing(dir) + '\\' + name;
    };

    const parent = (p: string): string => {
        if (!p) return '';
        const norm = normalize(p).replace(/[\\/]+$/, '');
        // 'C:' or 'C:\' (after stripping) → drive list view.
        if (/^[A-Za-z]:$/.test(norm)) return '';
        const idx = norm.lastIndexOf('\\');
        if (idx < 0) return '';
        // 'C:\foo' → 'C:\'. Cut after the drive separator to preserve the root sep.
        if (idx === 2 && /^[A-Za-z]:$/.test(norm.substring(0, 2))) {
            return norm.substring(0, 3); // 'C:\'
        }
        return norm.substring(0, idx);
    };

    const basename = (p: string): string => {
        const norm = normalize(p).replace(/[\\/]+$/, '');
        const idx = norm.lastIndexOf('\\');
        return idx >= 0 ? norm.substring(idx + 1) : norm;
    };

    const splitCrumbs = (p: string): Crumb[] => {
        if (!p) return [];
        const norm = normalize(p).replace(/[\\/]+$/, '');
        const match = norm.match(/^([A-Za-z]:)(.*)$/);
        if (!match) return [];
        const drive = match[1];                      // 'C:'
        const tail = match[2].replace(/^\\+/, '');   // 'Users\skm'
        const parts = tail ? tail.split('\\').filter(Boolean) : [];
        const crumbs: Crumb[] = [{ segment: drive, fullPath: drive + '\\' }];
        let acc = drive;
        for (const part of parts) {
            acc += '\\' + part;
            crumbs.push({ segment: part, fullPath: acc });
        }
        return crumbs;
    };

    const resolveTilde = (p: string): string => {
        if (!p) return p;
        if (p === '~') return info.home;
        if (p.startsWith('~\\') || p.startsWith('~/')) {
            return normalize(info.home + p.slice(1));
        }
        return p;
    };

    return {
        ...info,
        isRoot,
        isAbsolute,
        normalize,
        join,
        parent,
        basename,
        splitCrumbs,
        ensureTrailing,
        stripTrailing,
        resolveTilde,
        samplePath: 'C:\\path\\to\\videos',
    };
}

function makePosixAdapter(info: PlatformInfo): PathAdapter {
    const normalize = (p: string): string => {
        if (!p) return p;
        // Don't merge a leading '\' into '/'; on POSIX it's a literal char.
        // Just dedupe forward slashes.
        return p.replace(/\/+/g, '/');
    };
    const isAbsolute = (p: string): boolean => p.startsWith('/');
    const isRoot = (p: string): boolean => p === '/';
    const stripTrailing = (p: string): string => {
        if (!p || p === '/') return p;
        return p.replace(/\/+$/, '') || '/';
    };
    const ensureTrailing = (p: string): string => {
        if (!p) return p;
        return p.endsWith('/') ? p : p + '/';
    };
    const join = (dir: string, name: string): string => {
        if (!dir) return name;
        if (dir === '/') return '/' + name;
        return stripTrailing(dir) + '/' + name;
    };
    const parent = (p: string): string => {
        if (!p || p === '/') return '/';
        const trimmed = p.replace(/\/+$/, '');
        const idx = trimmed.lastIndexOf('/');
        if (idx <= 0) return '/';
        return trimmed.substring(0, idx);
    };
    const basename = (p: string): string => {
        const trimmed = p.replace(/\/+$/, '');
        const idx = trimmed.lastIndexOf('/');
        return idx >= 0 ? trimmed.substring(idx + 1) : trimmed;
    };
    const splitCrumbs = (p: string): Crumb[] => {
        if (!p || p === '/') return [{ segment: '/', fullPath: '/' }];
        const parts = p.replace(/\/+$/, '').split('/').filter(Boolean);
        const crumbs: Crumb[] = [{ segment: '/', fullPath: '/' }];
        let acc = '';
        for (const part of parts) {
            acc += '/' + part;
            crumbs.push({ segment: part, fullPath: acc });
        }
        return crumbs;
    };
    const resolveTilde = (p: string): string => {
        if (!p) return p;
        if (p === '~') return info.home;
        if (p.startsWith('~/')) return info.home + p.slice(1);
        return p;
    };
    return {
        ...info,
        isRoot,
        isAbsolute,
        normalize,
        join,
        parent,
        basename,
        splitCrumbs,
        ensureTrailing,
        stripTrailing,
        resolveTilde,
        samplePath: '/path/to/videos',
    };
}

export function createPathAdapter(info: PlatformInfo): PathAdapter {
    return info.os === 'windows' ? makeWindowsAdapter(info) : makePosixAdapter(info);
}

export const DEFAULT_POSIX_PLATFORM: PlatformInfo = {
    os: 'posix',
    sep: '/',
    home: '/',
    roots: [{ path: '/', label: '/', volumeName: '', free: null, total: null }],
};

export const DEFAULT_PATH_ADAPTER: PathAdapter = createPathAdapter(DEFAULT_POSIX_PLATFORM);
