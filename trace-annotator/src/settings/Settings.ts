import {PopupWindowType} from '../data/enums/PopupWindowType';

export class Settings {
    public static readonly TOP_NAVIGATION_BAR_HEIGHT_PX: number = 48;
    public static readonly EDITOR_BOTTOM_NAVIGATION_BAR_HEIGHT_PX: number = 240;
    public static readonly FILE_BROWSER_WIDTH_PX: number = 280;
    public static readonly FILE_BROWSER_MIN_WIDTH_PX: number = 260;
    public static readonly FILE_BROWSER_MAX_WIDTH_PX: number = 460;
    public static readonly FILE_BROWSER_COLLAPSED_WIDTH_PX: number = 24;
    public static readonly FILE_BROWSER_RESIZE_HANDLE_WIDTH_PX: number = 10;
    public static readonly BEHAVIOR_BAR_HEIGHT_PX: number = 36;
    public static readonly CLIPS_TABLE_MAX_HEIGHT_PX: number = 160;
    public static readonly TOOLKIT_TAB_HEIGHT_PX: number = 40;
    public static readonly TOOLBOX_PANEL_WIDTH_PX: number = 50 + 1;
    public static readonly MAX_DROPDOWN_OPTION_LENGTH: number = 20;

    public static readonly EDITOR_MIN_WIDTH: number = 900;
    public static readonly EDITOR_MIN_HEIGHT: number = 500;

    public static readonly PRIMARY_COLOR: string = '#009efd';
    public static readonly SECONDARY_COLOR: string = '#009efd';

    public static readonly DARK_THEME_FIRST_COLOR: string = '#0f1114';
    public static readonly DARK_THEME_SECOND_COLOR: string = '#1a1d23';
    public static readonly DARK_THEME_THIRD_COLOR: string = '#2e3340';
    public static readonly DARK_THEME_FORTH_COLOR: string = '#242830';

    public static readonly CLOSEABLE_POPUPS: PopupWindowType[] = [
        PopupWindowType.IMPORT_IMAGES,
        PopupWindowType.EXPORT_ANNOTATIONS,
        PopupWindowType.IMPORT_ANNOTATIONS,
        PopupWindowType.EXIT_PROJECT,
        PopupWindowType.UPDATE_LABEL
    ];

    public static readonly LABEL_COLORS_PALETTE = [
        '#ff3838',
        '#ff9d97',
        '#ff701f',
        '#ffb21d',
        '#cff231',
        '#48f90a',
        '#92cc17',
        '#3ddb86',
        '#1a9334',
        '#00d4bb',
        '#2c99a8',
        '#00c2ff',
        '#344593',
        '#6473ff',
        '#0018ec',
        '#8438ff',
        '#520085',
        '#cb38ff',
        '#ff95c8',
        '#ff37c7'
    ];

    public static readonly CSV_SEPARATOR = ','
    public static readonly RECT_LABELS_EXPORT_CSV_COLUMN_NAMES = [
        'image_name',
        'start_frame',
        'end_frame',
        'animal',
        'behavior',
    ].join(Settings.CSV_SEPARATOR);
}
