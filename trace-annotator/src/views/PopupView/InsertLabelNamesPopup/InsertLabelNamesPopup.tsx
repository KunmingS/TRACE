import React, { useState, useEffect, useMemo, useRef } from 'react';
import './InsertLabelNamesPopup.scss';
import { consumePendingFocusField, consumePendingFocusLabelId } from './focusState';
import { GenericYesNoPopup } from '../GenericYesNoPopup/GenericYesNoPopup';
import { PopupWindowType } from '../../../data/enums/PopupWindowType';
import { updateLabelNames, updateSubjects, updateActiveSubjectId } from '../../../store/labels/actionCreators';
import { updateActivePopupType, updatePerClassColorationStatus } from '../../../store/general/actionCreators';
import { AppState } from '../../../store';
import { connect } from 'react-redux';
import Scrollbars from 'react-custom-scrollbars-2';
import { LabelName, Subject } from '../../../store/labels/types';
import { LabelUtil } from '../../../utils/LabelUtil';
import { LabelsSelector } from '../../../store/selectors/LabelsSelector';
import { LabelActions } from '../../../logic/actions/LabelActions';
import { Settings } from '../../../settings/Settings';
import { reject, sample, uniq } from 'lodash';
import { submitNewNotification } from '../../../store/notifications/actionCreators';
import { INotification } from '../../../store/notifications/types';
import { NotificationUtil } from '../../../utils/NotificationUtil';
import { NotificationsDataMap } from '../../../data/info/NotificationsData';
import { Notification } from '../../../data/enums/Notification';
import { DEFAULT_SUBJECT_ID } from '../../../store/labels/reducer';

interface IProps {
    updateActivePopupTypeAction: (activePopupType: PopupWindowType) => any;
    updateLabelNamesAction: (labels: LabelName[]) => any;
    updateSubjectsAction: (subjects: Subject[]) => any;
    updateActiveSubjectIdAction: (id: string | null) => any;
    updatePerClassColorationStatusAction: (updatePerClassColoration: boolean) => any;
    submitNewNotificationAction: (notification: INotification) => any;
    isUpdate: boolean;
    enablePerClassColoration: boolean;
}

// Researcher-flavoured starter taxonomy — kept as an always-visible quick-add
// strip so users can seed multiple defaults without the list resetting the UI.
const STARTER_BEHAVIOURS = ['Eating', 'Resting', 'Grooming', 'Walking', 'Sleeping'];

const InsertLabelNamesPopup: React.FC<IProps> = (
    {
        updateActivePopupTypeAction,
        updateLabelNamesAction,
        updateSubjectsAction,
        updateActiveSubjectIdAction,
        updatePerClassColorationStatusAction,
        submitNewNotificationAction,
        isUpdate,
        enablePerClassColoration
    }) => {
    const [labelNames, setLabelNames] = useState(LabelsSelector.getLabelNames());
    const [subjects, setSubjects] = useState<Subject[]>(LabelsSelector.getSubjects());
    const [lastAddedId, setLastAddedId] = useState<string | null>(null);
    const [autoFocusField, setAutoFocusField] = useState<'name' | 'shortcut'>('name');
    const [focusedLabelId, setFocusedLabelId] = useState<string | null>(() => consumePendingFocusLabelId());
    const [focusedField] = useState(() => consumePendingFocusField());
    const rowRefs = useRef<Record<string, HTMLDivElement | null>>({});

    // Count clips per existing subject id (across all videos) to drive both
    // the "X clips" hint and the deletion guard.
    const clipCountByAnimalId = useMemo(() => {
        const counts = new Map<string, number>();
        for (const img of LabelsSelector.getImagesData()) {
            for (const rect of img.labelRects || []) {
                const id = rect.animalId ?? DEFAULT_SUBJECT_ID;
                counts.set(id, (counts.get(id) || 0) + 1);
            }
        }
        return counts;
    }, []);

    useEffect(() => {
        if (!focusedLabelId) return undefined;
        const node = rowRefs.current[focusedLabelId];
        if (node) {
            node.scrollIntoView({block: 'nearest', behavior: 'smooth'});
            const selector = focusedField === 'shortcut' ? '.ShortcutInput' : '.NameInput';
            const input = node.querySelector<HTMLInputElement>(selector);
            input?.focus();
            input?.select();
        }
        const t = window.setTimeout(() => setFocusedLabelId(null), 1600);
        return () => window.clearTimeout(t);
    }, [focusedLabelId, focusedField]);

    // ── Validation ─────────────────────────────────────────────────────
    const trimmedNames = labelNames.map(l => l.name.trim());
    const shortcuts = labelNames.map(l => l.shortcut || '');
    const nonEmptyShortcuts = shortcuts.filter(Boolean);

    const hasEmptyName = trimmedNames.some(n => n.length === 0);
    const hasDuplicateName = uniq(trimmedNames.map(n => n.toLowerCase())).length !== trimmedNames.length;
    const hasEmptyShortcut = shortcuts.some(s => !s);
    const hasDuplicateShortcut = uniq(nonEmptyShortcuts).length !== nonEmptyShortcuts.length;

    const trimmedSubjectNames = subjects.map(s => s.name.trim());
    const hasEmptySubjectName = trimmedSubjectNames.some(n => n.length === 0);
    const hasDuplicateSubjectName = uniq(trimmedSubjectNames.map(n => n.toLowerCase())).length !== trimmedSubjectNames.length;
    const newSubjectIds = new Set(subjects.map(s => s.id));
    const removedSubjectsWithClips = LabelsSelector.getSubjects()
        .filter(s => !newSubjectIds.has(s.id))
        .filter(s => (clipCountByAnimalId.get(s.id) || 0) > 0);
    const hasNoSubjects = subjects.length === 0;

    let validationHint: string | null = null;
    if (hasNoSubjects) {
        validationHint = 'You need at least one subject (animal).';
    } else if (hasEmptySubjectName) {
        validationHint = 'Name every subject before saving.';
    } else if (hasDuplicateSubjectName) {
        validationHint = 'Subject names must be unique.';
    } else if (removedSubjectsWithClips.length > 0) {
        const s = removedSubjectsWithClips[0];
        const n = clipCountByAnimalId.get(s.id) || 0;
        validationHint = `${s.name} has ${n} clip${n === 1 ? '' : 's'} — delete them first.`;
    } else if (labelNames.length > 0) {
        if (hasEmptyName) validationHint = 'Name every behavior before you start.';
        else if (hasDuplicateName) validationHint = 'Behavior names must be unique.';
        else if (hasEmptyShortcut) validationHint = 'Every behavior needs a one-key shortcut.';
        else if (hasDuplicateShortcut) validationHint = 'Shortcut keys must be unique.';
    }
    const canAccept = validationHint === null;

    const emitError = (n: Notification) => submitNewNotificationAction(
        NotificationUtil.createErrorNotification(NotificationsDataMap[n])
    );

    // ── List mutations ─────────────────────────────────────────────────
    const addLabelNameCallback = (seedName = '', focus: 'name' | 'shortcut' = 'name') => {
        const next = LabelUtil.createLabelName(seedName);
        setLabelNames([...labelNames, next]);
        setLastAddedId(next.id);
        setAutoFocusField(focus);
    };

    const handleAddClick = () => {
        // Block adding a new blank row while existing rows still have empty names.
        if (hasEmptyName && labelNames.length > 0) {
            emitError(Notification.EMPTY_LABEL_NAME_ERROR);
            return;
        }
        if (hasDuplicateName) {
            emitError(Notification.NON_UNIQUE_LABEL_NAMES_ERROR);
            return;
        }
        addLabelNameCallback();
    };

    const seedFromSuggestion = (name: string) => {
        if (labelNames.some(l => l.name.trim().toLowerCase() === name.toLowerCase())) return;
        addLabelNameCallback(name, 'shortcut');
    };

    const availableSuggestions = STARTER_BEHAVIOURS.filter(
        s => !labelNames.some(l => l.name.trim().toLowerCase() === s.toLowerCase())
    );

    const deleteLabelNameCallback = (id: string) => {
        setLabelNames(reject(labelNames, { id }));
    };

    const togglePerClassColorationCallback = () => {
        updatePerClassColorationStatusAction(!enablePerClassColoration);
    };

    const changeLabelNameColorCallback = (id: string) => {
        setLabelNames(labelNames.map((labelName: LabelName) => (
            labelName.id === id
                ? { ...labelName, color: sample(Settings.LABEL_COLORS_PALETTE) }
                : labelName
        )));
    };

    const onKeyUpCallback = (event: React.KeyboardEvent<HTMLInputElement>) => {
        if (event.key === 'Enter') handleAddClick();
    };

    const onChange = (id: string, value: string) => {
        setLabelNames(labelNames.map((labelName: LabelName) => (
            labelName.id === id ? { ...labelName, name: value } : labelName
        )));
    };

    const onChangeShortcut = (id: string, value: string) => {
        const shortcut = value.toLowerCase().replace(/[^a-z]/g, '').slice(0, 1);
        setLabelNames(labelNames.map((labelName: LabelName) => (
            labelName.id === id ? { ...labelName, shortcut } : labelName
        )));
    };

    // ── Subject mutations ──────────────────────────────────────────────
    const addSubjectCallback = () => {
        const next = LabelUtil.createSubject(`Animal ${subjects.length + 1}`);
        setSubjects([...subjects, next]);
    };

    const deleteSubjectCallback = (id: string) => {
        setSubjects(reject(subjects, { id }));
    };

    const onChangeSubjectName = (id: string, value: string) => {
        setSubjects(subjects.map(s => s.id === id ? { ...s, name: value } : s));
    };

    // ── Accept/reject ──────────────────────────────────────────────────
    const persistSubjects = () => {
        updateSubjectsAction(subjects);
        const activeId = LabelsSelector.getActiveSubjectId();
        if (!subjects.some(s => s.id === activeId)) {
            updateActiveSubjectIdAction(subjects[0]?.id ?? null);
        }
    };

    const onCreateAcceptCallback = () => {
        persistSubjects();
        updateLabelNamesAction(labelNames);
        updateActivePopupTypeAction(null);
    };

    const onUpdateAcceptCallback = () => {
        const missingIds: string[] = LabelUtil.labelNamesIdsDiff(LabelsSelector.getLabelNames(), labelNames);
        LabelActions.removeLabelNames(missingIds);
        persistSubjects();
        updateLabelNamesAction(labelNames);
        updateActivePopupTypeAction(null);
    };

    const onUpdateRejectCallback = () => {
        updateActivePopupTypeAction(null);
    };

    const hasLabels = labelNames.length > 0;

    // ── Render helpers ─────────────────────────────────────────────────
    const renderSuggestionStrip = (compact: boolean) => {
        if (availableSuggestions.length === 0) return null;
        return (
            <div className={`SuggestionChips ${compact ? 'compact' : ''}`}>
                {!compact && <span className='SuggestionLabel'>Quick add</span>}
                {compact && <span className='SuggestionLabel'>More:</span>}
                {availableSuggestions.map(name => (
                    <button
                        key={name}
                        type='button'
                        className='SuggestionChip'
                        onClick={() => seedFromSuggestion(name)}
                    >
                        + {name}
                    </button>
                ))}
            </div>
        );
    };

    const renderRow = (labelName: LabelName, index: number) => {
        const isLastAdded = labelName.id === lastAddedId;
        const focusName = isLastAdded && autoFocusField === 'name';
        const focusShortcut = isLastAdded && autoFocusField === 'shortcut';
        const rowShortcut = labelName.shortcut || '';
        const shortcutDuplicate = rowShortcut !== '' && shortcuts.filter(s => s === rowShortcut).length > 1;
        const nameDuplicate = labelName.name.trim() !== '' && trimmedNames
            .map(n => n.toLowerCase())
            .filter(n => n === labelName.name.trim().toLowerCase()).length > 1;
        const isFocused = labelName.id === focusedLabelId;
        return (
            <div
                className={`BehaviorRow ${nameDuplicate || shortcutDuplicate ? 'has-error' : ''} ${isFocused ? 'is-focused' : ''}`}
                key={labelName.id}
                ref={(el) => { rowRefs.current[labelName.id] = el; }}
            >
                <span className='RowIndex'>{String(index + 1).padStart(2, '0')}</span>
                <button
                    type='button'
                    className='ColorChip'
                    style={{ backgroundColor: labelName.color || '#cbd5e1' }}
                    onClick={() => changeLabelNameColorCallback(labelName.id)}
                    aria-label='Shuffle color'
                    title='Click to shuffle color'
                />
                <input
                    className={`NameInput ${nameDuplicate ? 'error' : ''}`}
                    type='text'
                    autoComplete='off'
                    autoFocus={focusName}
                    placeholder='Behavior name'
                    value={labelName.name}
                    onChange={(e) => onChange(labelName.id, e.target.value)}
                    onKeyUp={onKeyUpCallback}
                />
                <div className={`ShortcutField ${rowShortcut === '' ? 'missing' : ''} ${shortcutDuplicate ? 'error' : ''}`}>
                    <span className='ShortcutHint'>key</span>
                    <input
                        className='ShortcutInput'
                        type='text'
                        autoComplete='off'
                        autoFocus={focusShortcut}
                        placeholder='—'
                        maxLength={1}
                        value={rowShortcut}
                        onChange={(e) => onChangeShortcut(labelName.id, e.target.value)}
                    />
                </div>
                <button
                    type='button'
                    className='RowDeleteBtn'
                    aria-label='Remove behavior'
                    onClick={() => deleteLabelNameCallback(labelName.id)}
                >
                    <svg width='14' height='14' viewBox='0 0 14 14' fill='none'>
                        <path d='M3 3l8 8M11 3l-8 8' stroke='currentColor' strokeWidth='1.4' strokeLinecap='round'/>
                    </svg>
                </button>
            </div>
        );
    };

    const renderSubjectRow = (subject: Subject, index: number) => {
        const clipCount = clipCountByAnimalId.get(subject.id) || 0;
        const nameTrimmed = subject.name.trim();
        const dup = nameTrimmed !== '' && trimmedSubjectNames
            .map(n => n.toLowerCase())
            .filter(n => n === nameTrimmed.toLowerCase()).length > 1;
        return (
            <div className={`SubjectRow ${dup ? 'has-error' : ''}`} key={subject.id}>
                <span className='SubjectDigit' aria-hidden='true'>{index + 1}</span>
                <input
                    className={`SubjectNameInput ${dup ? 'error' : ''}`}
                    type='text'
                    autoComplete='off'
                    placeholder='Subject name'
                    value={subject.name}
                    onChange={(e) => onChangeSubjectName(subject.id, e.target.value)}
                />
                {clipCount > 0 && (
                    <span className='SubjectClipCount' title={`${clipCount} clip${clipCount === 1 ? '' : 's'}`}>
                        {clipCount} clip{clipCount === 1 ? '' : 's'}
                    </span>
                )}
                <button
                    type='button'
                    className='RowDeleteBtn'
                    aria-label='Remove subject'
                    onClick={() => deleteSubjectCallback(subject.id)}
                >
                    <svg width='14' height='14' viewBox='0 0 14 14' fill='none'>
                        <path d='M3 3l8 8M11 3l-8 8' stroke='currentColor' strokeWidth='1.4' strokeLinecap='round'/>
                    </svg>
                </button>
            </div>
        );
    };

    const renderContent = () => (
        <div className='BehaviorTypesPopup'>
            <div className='Intro'>
                <p className='IntroText'>
                    {isUpdate
                        ? 'Rename, recolor, or remove the behaviors used in this project. Each one gets a one-key shortcut for fast tagging.'
                        : 'Name the behaviors you\'ll annotate and assign each a one-key shortcut for fast tagging. You can also start with no behaviors and add them as you go.'}
                </p>
            </div>

            <div className='SubjectsPanel'>
                <div className='PanelHeader'>
                    <span className='PanelTitle'>
                        Subjects
                        <span className='PanelCount'>{subjects.length}</span>
                    </span>
                    <span className='PanelSubtitle'>Press 1–9 in the editor to switch</span>
                    <button
                        type='button'
                        className='AddBehaviorBtn'
                        onClick={addSubjectCallback}
                        aria-label='Add subject'
                        disabled={subjects.length >= 9}
                    >
                        <svg width='12' height='12' viewBox='0 0 12 12' fill='none'>
                            <path d='M6 2v8M2 6h8' stroke='currentColor' strokeWidth='1.6' strokeLinecap='round'/>
                        </svg>
                        Add subject
                    </button>
                </div>
                <div className='SubjectsList'>
                    {subjects.map(renderSubjectRow)}
                </div>
            </div>

            <div className='BehaviorsPanel'>
                <div className='PanelHeader'>
                    <span className='PanelTitle'>
                        Behavior types
                        {hasLabels && <span className='PanelCount'>{labelNames.length}</span>}
                    </span>
                    <button
                        type='button'
                        className='AddBehaviorBtn'
                        onClick={handleAddClick}
                        aria-label='Add behavior type'
                    >
                        <svg width='12' height='12' viewBox='0 0 12 12' fill='none'>
                            <path d='M6 2v8M2 6h8' stroke='currentColor' strokeWidth='1.6' strokeLinecap='round'/>
                        </svg>
                        Add behavior
                    </button>
                </div>

                {hasLabels ? (
                    <>
                        <div className='ListWrap'>
                            <Scrollbars autoHeight autoHeightMax={260}>
                                <div className='BehaviorList'>
                                    {labelNames.map((labelName, i) => renderRow(labelName, i))}
                                </div>
                            </Scrollbars>
                        </div>
                        {renderSuggestionStrip(true)}
                    </>
                ) : (
                    <div className='EmptyState'>
                        <div className='EmptyHeading'>No behaviors yet</div>
                        <div className='EmptyHint'>Click a suggestion to seed one, or define your own above.</div>
                        {renderSuggestionStrip(false)}
                    </div>
                )}

                <div className='PanelFooterHint'>
                    <label className='ColorToggle'>
                        <input
                            type='checkbox'
                            checked={enablePerClassColoration}
                            onChange={togglePerClassColorationCallback}
                        />
                        <span className='ColorToggleTrack'><span className='ColorToggleThumb'/></span>
                        <span className='ColorToggleLabel'>Per-class colors during annotation</span>
                    </label>
                    {validationHint && (
                        <span className='ValidationHint' role='status'>
                            <svg width='12' height='12' viewBox='0 0 12 12' fill='none'>
                                <circle cx='6' cy='6' r='5' stroke='currentColor' strokeWidth='1.2'/>
                                <path d='M6 3.5v2.8M6 8.2v.2' stroke='currentColor' strokeWidth='1.4' strokeLinecap='round'/>
                            </svg>
                            {validationHint}
                        </span>
                    )}
                </div>
            </div>
        </div>
    );

    return (
        <GenericYesNoPopup
            title={isUpdate ? 'Edit Behavior Types' : 'Create Behavior Types'}
            renderContent={renderContent}
            acceptLabel={isUpdate ? 'Save' : 'Start annotation'}
            onAccept={isUpdate ? onUpdateAcceptCallback : onCreateAcceptCallback}
            disableAcceptButton={!canAccept}
            rejectLabel={isUpdate ? 'Cancel' : undefined}
            onReject={isUpdate ? onUpdateRejectCallback : undefined}
            skipRejectButton={!isUpdate}
        />
    );
};

const mapDispatchToProps = {
    updateActivePopupTypeAction: updateActivePopupType,
    updateLabelNamesAction: updateLabelNames,
    updateSubjectsAction: updateSubjects,
    updateActiveSubjectIdAction: updateActiveSubjectId,
    updatePerClassColorationStatusAction: updatePerClassColorationStatus,
    submitNewNotificationAction: submitNewNotification
};

const mapStateToProps = (state: AppState) => ({
    enablePerClassColoration: state.general.enablePerClassColoration
});

export default connect(
    mapStateToProps,
    mapDispatchToProps
)(InsertLabelNamesPopup);
