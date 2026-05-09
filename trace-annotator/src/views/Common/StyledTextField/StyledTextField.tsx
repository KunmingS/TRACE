import { TextField } from '@mui/material';
import { styled } from '@mui/system';
import { Settings } from '../../../settings/Settings';


export const StyledTextField = styled(TextField)({
    '& .MuiInputBase-root': {
        color: 'rgba(0, 0, 0, 0.88)',
    },
    '& label': {
        color: 'rgba(0, 0, 0, 0.55)',
    },
    '& .MuiInput-underline:before': {
        borderBottomColor: 'rgba(0, 0, 0, 0.16)',
    },
    '& .MuiInput-underline:hover:before': {
        borderBottomColor: 'rgba(0, 0, 0, 0.32)',
    },
    '& label.Mui-focused': {
        color: Settings.SECONDARY_COLOR,
    },
    '& .MuiInput-underline:after': {
        borderBottomColor: Settings.SECONDARY_COLOR,
    }
});
