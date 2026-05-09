export type PendingFocusField = 'name' | 'shortcut';

let pendingFocusLabelId: string | null = null;
let pendingFocusField: PendingFocusField = 'name';

export const setPendingFocusLabelId = (id: string | null, field: PendingFocusField = 'name'): void => {
    pendingFocusLabelId = id;
    pendingFocusField = field;
};

export const consumePendingFocusLabelId = (): string | null => {
    const id = pendingFocusLabelId;
    pendingFocusLabelId = null;
    return id;
};

export const consumePendingFocusField = (): PendingFocusField => {
    const field = pendingFocusField;
    pendingFocusField = 'name';
    return field;
};
