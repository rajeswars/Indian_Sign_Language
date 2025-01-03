import React, { useState, useEffect, useRef } from "react";
import { Box, Button, TextField, Typography, Paper, Container, Grid, Avatar } from '@mui/material';
import VideocamIcon from '@mui/icons-material/Videocam';
import VideocamOffIcon from '@mui/icons-material/VideocamOff';
import { styled } from '@mui/system';

const StyledButton = styled(Button)(({ theme }) => ({
    '&:hover': {
        backgroundColor: theme.palette.secondary.light,
        transform: 'scale(1.05)',
        transition: 'transform 0.2s ease-in-out',
    },
}));

const StyledPaper = styled(Paper)(({ theme }) => ({
    padding: theme.spacing(4),
    textAlign: 'center',
    backgroundColor: '#f5f5f5',
    borderRadius: theme.shape.borderRadius,
    boxShadow: '0px 4px 12px rgba(0, 0, 0, 0.1)',
}));

const HandGestureRecognition = () => {
    const [sign, setSign] = useState("No gesture detected");
    const [message, setMessage] = useState(""); // New state for the message box
    const [isWebcamOn, setIsWebcamOn] = useState(false);
    const wsRef = useRef(null); // Ref to store WebSocket
    const prevGestureRef = useRef(""); // Ref for previous gesture

    useEffect(() => {
        if (isWebcamOn) {
            const videoFeedUrl = `http://localhost:5000/video_feed?t=${new Date().getTime()}`;
            document.getElementById("video-stream").src = videoFeedUrl;

            // Connect to WebSocket for gesture recognition
            const websocket = new WebSocket("ws://localhost:5000/ws");
            wsRef.current = websocket; // Store WebSocket in ref

            websocket.onopen = () => {
                console.log("WebSocket connected");
            };

            websocket.onerror = (event) => {
                console.error("WebSocket error observed:", event);
            };

            websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                console.log("Received gesture:", data.gesture);  // Debug log
                if (data && data.gesture !== undefined && data.gesture !== null) {
                    // Update detected sign
                    setSign(data.gesture.toString() || "No gesture detected");
                    // Update message box only if a valid gesture is detected
                    if (data.gesture !== "No gesture detected") {
                        if (prevGestureRef.current !== data.gesture) {
                            prevGestureRef.current = data.gesture; // Update ref with current gesture
                            if(data.gesture==="Space"){
                                data.gesture = " ";
                            }
                            setMessage((prevMessage) => prevMessage + data.gesture); // Append the current gesture
                        }
                    } else {
                        prevGestureRef.current = "";
                    }
                }
            };

            websocket.onclose = () => {
                console.log("WebSocket disconnected");
                wsRef.current = null; // Clear WebSocket ref on close
            };

            return () => {
                websocket.close(); // Close the WebSocket connection on cleanup
                wsRef.current = null; // Clear WebSocket ref on cleanup
            };
        }
    }, [isWebcamOn]);

    const toggleWebcam = () => {
        if (!isWebcamOn) {
            fetch("http://localhost:5000/start", { method: "POST" })
                .then(() => setIsWebcamOn(true))
                .catch((err) => console.error("Error starting webcam:", err));
        } else {
            fetch("http://localhost:5000/stop", { method: "POST" })
                .then(() => {
                    setIsWebcamOn(false);
                    document.getElementById("video-stream").src = ""; // Clear the video feed when webcam is off
                    setSign("No gesture detected"); // Reset sign to "No gesture detected" when stopping webcam
                    setMessage(""); // Clear the message box when stopping webcam
                    prevGestureRef.current = ""; // Reset the previous gesture ref
                })
                .catch((err) => console.error("Error stopping webcam:", err));
        }
    };

    return (
        <Container maxWidth="md" sx={{ mt: 4 }}>
            <StyledPaper elevation={3}>
                <Typography variant="h4" gutterBottom color="primary">
                    Hand Gesture Recognition
                </Typography>

                <StyledButton
                    variant="contained"
                    color={isWebcamOn ? "error" : "primary"}
                    startIcon={isWebcamOn ? <VideocamOffIcon /> : <VideocamIcon />}
                    onClick={toggleWebcam}
                    sx={{ mb: 3, fontWeight: 'bold' }}
                >
                    {isWebcamOn ? "Turn Off Webcam" : "Turn On Webcam"}
                </StyledButton>

                <Grid container spacing={4}>
                    {/* Webcam Feed */}
                    <Grid item xs={12} md={6}>
                        <Typography variant="h6" gutterBottom color="secondary">
                            Webcam Feed
                        </Typography>
                        <Box
                            component="img"
                            id="video-stream"
                            alt="Webcam feed"
                            width="100%"
                            height="auto"
                            sx={{
                                borderRadius: 2,
                                border: '3px solid #4caf50',
                                p: 1,
                                boxShadow: '0px 6px 10px rgba(0,0,0,0.1)',
                                transition: 'border-color 0.3s ease-in-out',
                                '&:hover': {
                                    borderColor: '#2196f3',
                                },
                            }}
                        />
                    </Grid>

                    {/* Detected Sign and Message Box */}
                    <Grid item xs={12} md={6}>
                        <Typography variant="h6" gutterBottom color="secondary">
                            Detected Sign
                        </Typography>
                        <TextField
                            fullWidth
                            value={sign}
                            variant="outlined"
                            inputProps={{ readOnly: true, style: { textAlign: 'center', fontSize: '1.5rem', color: '#ff5722' } }}
                            sx={{ mb: 3, backgroundColor: '#fff3e0', borderRadius: 1 }}
                        />

                        <Typography variant="h6" sx={{ mt: 4 }} gutterBottom color="secondary">
                            Message Box
                        </Typography>
                        <TextField
                            fullWidth
                            value={message}
                            variant="outlined"
                            inputProps={{ readOnly: true, style: { textAlign: 'center', fontSize: '1.2rem', color: '#3f51b5' } }}
                            multiline
                            rows={4}
                            sx={{ backgroundColor: '#e3f2fd', borderRadius: 1 }}
                        />
                    </Grid>
                </Grid>
            </StyledPaper>
        </Container>
    );
};

export default HandGestureRecognition;
