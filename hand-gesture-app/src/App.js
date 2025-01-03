import React from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import HandGestureRecognition from './HandGestureRecognition';

const theme = createTheme({
    palette: {
        mode: 'light', // or 'dark' if you prefer
        primary: {
            main: '#007bff',
        },
        secondary: {
            main: '#6c757d',
        },
        background: {
            default: '#f4f4f9',
            paper: '#ffffff',
        },
    },
});

function App() {
    return (
        <ThemeProvider theme={theme}>
            <CssBaseline /> {/* Ensures consistent baseline styles */}
            <div className="App">
                <HandGestureRecognition />
            </div>
        </ThemeProvider>
    );
}

export default App;
