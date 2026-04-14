export type DropDownMenuNode = {
    name: string;
    imageSrc: string;
    imageAlt: string;
    disabled?: boolean;
    onClick?: () => void;
    children?: DropDownMenuNode[];
}

export const DropDownMenuData: DropDownMenuNode[] = [
    {
        name: 'Actions',
        imageSrc: 'ico/actions.png',
        imageAlt: 'actions',
        children: []
    },
    {
        name: 'Export',
        imageSrc: 'ico/export.png',
        imageAlt: 'export',
        children: []
    }
];
