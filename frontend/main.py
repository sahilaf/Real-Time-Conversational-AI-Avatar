from flask import Flask, request, jsonify
from flask_cors import CORS
from livekit.api import AccessToken, VideoGrants
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# LiveKit configuration
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")

if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
    raise ValueError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set in .env file")


@app.route("/")
def index():
    """Serve the frontend HTML"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agents Playground</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/livekit-client/2.5.8/livekit-client.umd.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            min-height: 100vh;
        }

        .header {
            background: #000;
            border-bottom: 1px solid #1a1a1a;
            padding: 16px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header-title {
            font-size: 16px;
            font-weight: 500;
            color: #fff;
        }

        .connect-btn {
            background: #0ea5e9;
            color: white;
            border: none;
            padding: 8px 20px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
        }

        .connect-btn:hover {
            background: #0284c7;
        }

        .connect-btn:disabled {
            background: #334155;
            cursor: not-allowed;
        }

        .main-container {
            display: flex;
            height: calc(100vh - 57px);
        }

        .video-section {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #0a0a0a;
            border-right: 1px solid #1a1a1a;
        }

        .video-header {
            padding: 12px 20px;
            background: #0f0f0f;
            border-bottom: 1px solid #1a1a1a;
            font-size: 12px;
            font-weight: 600;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .video-wrapper {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #000;
            position: relative;
            overflow: hidden;
        }

        video {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .no-video-message {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #666;
            font-size: 14px;
            text-align: center;
            z-index: 2;
            pointer-events: none;
        }

        .chat-placeholder {
            color: #666;
            font-size: 14px;
            text-align: center;
        }

        .chat-section {
            width: 400px;
            background: #0a0a0a;
            border-right: 1px solid #1a1a1a;
            display: flex;
            flex-direction: column;
        }

        .chat-header {
            padding: 12px 20px;
            background: #0f0f0f;
            border-bottom: 1px solid #1a1a1a;
            font-size: 12px;
            font-weight: 600;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }

        .chat-message {
            margin-bottom: 16px;
        }

        .chat-sender {
            font-size: 11px;
            font-weight: 600;
            color: #888;
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .chat-bubble {
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            padding: 10px 12px;
            color: #e0e0e0;
            font-size: 13px;
            line-height: 1.5;
        }

        .chat-bubble.user {
            background: #0ea5e914;
            border-color: #0ea5e933;
        }

        .chat-bubble.agent {
            background: #10b98114;
            border-color: #10b98133;
        }

        .chat-bubble.interim {
            opacity: 0.6;
            font-style: italic;
        }

        .sidebar {
            width: 320px;
            background: #0a0a0a;
            overflow-y: auto;
            padding: 20px;
        }

        .sidebar-section {
            margin-bottom: 24px;
        }

        .section-title {
            font-size: 11px;
            font-weight: 600;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }

        .section-content {
            background: #0f0f0f;
            border: 1px solid #1a1a1a;
            border-radius: 6px;
            padding: 12px;
        }

        .info-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            font-size: 13px;
            border-bottom: 1px solid #1a1a1a;
        }

        .info-row:last-child {
            border-bottom: none;
        }

        .info-label {
            color: #888;
        }

        .info-value {
            color: #e0e0e0;
            font-weight: 500;
        }

        .status-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }

        .status-connected {
            background: #10b98114;
            color: #10b981;
        }

        .status-disconnected {
            background: #ef444414;
            color: #ef4444;
        }

        .status-connecting {
            background: #f59e0b14;
            color: #f59e0b;
        }

        .controls {
            padding: 16px;
            background: #0f0f0f;
            border-top: 1px solid #1a1a1a;
            display: flex;
            gap: 12px;
        }

        .control-button {
            flex: 1;
            padding: 10px;
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 6px;
            color: #e0e0e0;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .control-button:hover {
            background: #2a2a2a;
            border-color: #3a3a3a;
        }

        .control-button.active {
            background: #0ea5e9;
            border-color: #0ea5e9;
        }

        .control-button.muted {
            background: #dc2626;
            border-color: #dc2626;
        }

        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }

        .modal.active {
            display: flex;
        }

        .modal-content {
            background: #0f0f0f;
            border: 1px solid #2a2a2a;
            border-radius: 12px;
            padding: 32px;
            max-width: 450px;
            width: 90%;
        }

        .modal-title {
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 8px;
            color: #fff;
        }

        .modal-description {
            color: #888;
            font-size: 14px;
            margin-bottom: 24px;
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            font-size: 13px;
            font-weight: 500;
            color: #e0e0e0;
        }

        input {
            width: 100%;
            padding: 10px 12px;
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 6px;
            color: #e0e0e0;
            font-size: 14px;
            transition: border-color 0.2s;
        }

        input:focus {
            outline: none;
            border-color: #0ea5e9;
        }

        .modal-buttons {
            display: flex;
            gap: 12px;
            margin-top: 24px;
        }

        .btn {
            flex: 1;
            padding: 10px;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-primary {
            background: #0ea5e9;
            color: white;
        }

        .btn-primary:hover {
            background: #0284c7;
        }

        .btn-secondary {
            background: #1a1a1a;
            color: #e0e0e0;
            border: 1px solid #2a2a2a;
        }

        .btn-secondary:hover {
            background: #2a2a2a;
        }

        .audio-indicator {
            padding: 12px;
            background: #0f0f0f;
            border-radius: 6px;
            margin-top: 12px;
        }

        .audio-level-label {
            font-size: 12px;
            color: #888;
            margin-bottom: 8px;
        }

        .audio-level {
            height: 6px;
            background: #1a1a1a;
            border-radius: 3px;
            overflow: hidden;
        }

        .audio-level-bar {
            height: 100%;
            background: #10b981;
            width: 0%;
            transition: width 0.1s;
        }

        audio {
            display: none;
        }

        @media (max-width: 1024px) {
            .sidebar {
                width: 280px;
            }

            .chat-section {
                width: 350px;
            }
        }

        @media (max-width: 768px) {
            .main-container {
                flex-direction: column;
            }

            .video-section {
                border-right: none;
                border-bottom: 1px solid #1a1a1a;
                height: 50vh;
            }

            .chat-section {
                width: 100%;
                border-right: none;
                border-bottom: 1px solid #1a1a1a;
                height: 30vh;
            }

            .sidebar {
                width: 100%;
                height: 20vh;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-title">Agents Playground</div>
        <button class="connect-btn" id="connectBtn">Connect</button>
    </div>

    <div class="main-container">
        <div class="video-section">
            <div class="video-header">Agent Video</div>
            <div class="video-wrapper">
                <video id="remoteVideo" autoplay playsinline></video>
                <div id="noVideoMessage" class="no-video-message">
                    No agent video track. Connect to get started.
                </div>
            </div>
            <div class="controls">
                <button class="control-button" id="muteBtn">
                    <span>🎤</span>
                    <span>Mute</span>
                </button>
                <button class="control-button" id="disconnectBtn" style="display: none;">
                    Disconnect
                </button>
            </div>
        </div>

        <div class="chat-section">
            <div class="chat-header">Transcript</div>
            <div class="chat-messages" id="chatMessages">
                <div class="chat-placeholder">Transcriptions will appear here</div>
            </div>
        </div>

        <div class="sidebar">
            <div class="sidebar-section">
                <div class="section-title">Description</div>
                <div class="section-content">
                    <p style="color: #888; font-size: 13px; line-height: 1.5;">
                        A virtual workbench for testing multimodal AI agents.
                    </p>
                </div>
            </div>

            <div class="sidebar-section">
                <div class="section-title">Room</div>
                <div class="section-content">
                    <div class="info-row">
                        <span class="info-label">Room name</span>
                        <span class="info-value" id="roomNameDisplay">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Status</span>
                        <span id="statusBadge" class="status-badge status-disconnected">Disconnected</span>
                    </div>
                </div>
            </div>

            <div class="sidebar-section">
                <div class="section-title">Agent</div>
                <div class="section-content">
                    <div class="info-row">
                        <span class="info-label">Agent name</span>
                        <span class="info-value" id="agentNameDisplay">None</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Identity</span>
                        <span class="info-value" id="agentIdentityDisplay">No agent connected</span>
                    </div>
                </div>
            </div>

            <div class="sidebar-section">
                <div class="section-title">User</div>
                <div class="section-content">
                    <div class="info-row">
                        <span class="info-label">Name</span>
                        <span class="info-value" id="userNameDisplay">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Identity</span>
                        <span class="info-value" id="userIdentityDisplay">-</span>
                    </div>
                </div>

                <div class="audio-indicator">
                    <div class="audio-level-label">🎤 Your microphone level</div>
                    <div class="audio-level">
                        <div class="audio-level-bar" id="audioLevelBar"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Connection Modal -->
    <div class="modal" id="connectionModal">
        <div class="modal-content">
            <div class="modal-title">Connect to Room</div>
            <div class="modal-description">
                Enter your details to join the AI avatar session
            </div>
            <div class="form-group">
                <label for="roomName">Room Name</label>
                <input type="text" id="roomName" placeholder="my-room" value="my-room">
            </div>
            <div class="form-group">
                <label for="userName">Your Name</label>
                <input type="text" id="userName" placeholder="Enter your name" value="User">
            </div>
            <div class="modal-buttons">
                <button class="btn btn-secondary" id="cancelBtn">Cancel</button>
                <button class="btn btn-primary" id="confirmConnectBtn">Connect</button>
            </div>
        </div>
    </div>

    <!-- Hidden audio container -->
    <div id="audioContainer" style="display: none;"></div>

    <script>
        const LiveKit = window.LivekitClient;
        
        let room = null;
        let audioTrack = null;
        let isMuted = false;
        let currentUserTranscript = null;
        let currentAgentTranscript = null;

        const elements = {
            roomName: document.getElementById('roomName'),
            userName: document.getElementById('userName'),
            connectBtn: document.getElementById('connectBtn'),
            confirmConnectBtn: document.getElementById('confirmConnectBtn'),
            cancelBtn: document.getElementById('cancelBtn'),
            disconnectBtn: document.getElementById('disconnectBtn'),
            muteBtn: document.getElementById('muteBtn'),
            connectionModal: document.getElementById('connectionModal'),
            remoteVideo: document.getElementById('remoteVideo'),
            noVideoMessage: document.getElementById('noVideoMessage'),
            audioContainer: document.getElementById('audioContainer'),
            audioLevelBar: document.getElementById('audioLevelBar'),
            statusBadge: document.getElementById('statusBadge'),
            roomNameDisplay: document.getElementById('roomNameDisplay'),
            userNameDisplay: document.getElementById('userNameDisplay'),
            userIdentityDisplay: document.getElementById('userIdentityDisplay'),
            agentNameDisplay: document.getElementById('agentNameDisplay'),
            agentIdentityDisplay: document.getElementById('agentIdentityDisplay'),
            chatMessages: document.getElementById('chatMessages')
        };

        function addOrUpdateTranscript(sender, text, isFinal, isAgent = false) {
            const container = elements.chatMessages;
            const placeholder = container.querySelector('.chat-placeholder');
            if (placeholder) placeholder.remove();

            // Determine which transcript ID to use
            const transcriptId = isAgent ? 'agent-transcript' : 'user-transcript';
            let messageDiv = document.getElementById(transcriptId);

            if (isFinal) {
                // Create new final transcript
                messageDiv = document.createElement('div');
                messageDiv.className = 'chat-message';
                messageDiv.id = transcriptId + '-' + Date.now(); // Unique ID for final messages

                const senderDiv = document.createElement('div');
                senderDiv.className = 'chat-sender';
                senderDiv.textContent = sender;

                const bubble = document.createElement('div');
                bubble.className = `chat-bubble ${isAgent ? 'agent' : 'user'}`;
                bubble.textContent = text;

                messageDiv.appendChild(senderDiv);
                messageDiv.appendChild(bubble);
                container.appendChild(messageDiv);

                // Clear the interim reference
                if (isAgent) {
                    currentAgentTranscript = null;
                } else {
                    currentUserTranscript = null;
                }
            } else {
                // Update or create interim transcript
                if (!messageDiv) {
                    messageDiv = document.createElement('div');
                    messageDiv.className = 'chat-message';
                    messageDiv.id = transcriptId;

                    const senderDiv = document.createElement('div');
                    senderDiv.className = 'chat-sender';
                    senderDiv.textContent = sender;

                    const bubble = document.createElement('div');
                    bubble.className = `chat-bubble ${isAgent ? 'agent' : 'user'} interim`;
                    bubble.textContent = text;

                    messageDiv.appendChild(senderDiv);
                    messageDiv.appendChild(bubble);
                    container.appendChild(messageDiv);

                    if (isAgent) {
                        currentAgentTranscript = messageDiv;
                    } else {
                        currentUserTranscript = messageDiv;
                    }
                } else {
                    // Update existing interim transcript
                    const bubble = messageDiv.querySelector('.chat-bubble');
                    if (bubble) {
                        bubble.textContent = text;
                    }
                }
            }

            container.scrollTop = container.scrollHeight;
        }

        function handleTranscriptionReceived(segments, participant, publication) {
            console.log('Transcription received:', segments, 'from', participant?.identity);
            
            if (!segments || segments.length === 0) return;

            const isAgent = participant && (
                participant.identity.includes('agent') || 
                participant.identity.includes('Obama') ||
                participant.identity.startsWith('agent-')
            );
            
            const sender = isAgent ? 
                (elements.agentNameDisplay.textContent !== 'None' ? elements.agentNameDisplay.textContent : 'Agent') :
                (elements.userNameDisplay.textContent !== '-' ? elements.userNameDisplay.textContent : 'You');

            // Combine all segments
            const fullText = segments.map(s => s.text).join(' ');
            const isFinal = segments.some(s => s.final);

            if (fullText.trim()) {
                addOrUpdateTranscript(sender, fullText, isFinal, isAgent);
            }
        }

        function showModal() {
            elements.connectionModal.classList.add('active');
        }

        function hideModal() {
            elements.connectionModal.classList.remove('active');
        }

        function updateStatus(status, text) {
            elements.statusBadge.className = `status-badge status-${status}`;
            elements.statusBadge.textContent = text;
        }

        async function getToken(roomName, userName) {
            const response = await fetch('/token', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    room_name: roomName,
                    participant_name: userName
                })
            });

            if (!response.ok) {
                throw new Error('Failed to get token');
            }

            const data = await response.json();
            return data;
        }

        async function connect() {
            const roomName = elements.roomName.value.trim();
            const userName = elements.userName.value.trim();

            if (!roomName || !userName) {
                alert('Please enter both room name and your name');
                return;
            }

            try {
                updateStatus('connecting', 'Connecting');
                elements.confirmConnectBtn.disabled = true;
                elements.connectBtn.disabled = true;

                // Get token from server
                const { token, url } = await getToken(roomName, userName);

                room = new LiveKit.Room({
                    adaptiveStream: true,
                    dynacast: true,
                    videoCaptureDefaults: {
                        resolution: LiveKit.VideoPresets.h720.resolution,
                    },
                });

                // Set up event listeners
                room.on(LiveKit.RoomEvent.TrackSubscribed, handleTrackSubscribed);
                room.on(LiveKit.RoomEvent.TrackUnsubscribed, handleTrackUnsubscribed);
                room.on(LiveKit.RoomEvent.Disconnected, handleDisconnect);
                room.on(LiveKit.RoomEvent.LocalTrackPublished, handleLocalTrackPublished);
                room.on(LiveKit.RoomEvent.ParticipantConnected, handleParticipantConnected);
                room.on(LiveKit.RoomEvent.TranscriptionReceived, handleTranscriptionReceived);
                
                // Add click handler to resume audio context
                const resumeAudio = () => {
                    document.querySelectorAll('video, audio').forEach(el => {
                        el.play().catch(() => {});
                    });
                };
                document.addEventListener('click', resumeAudio, { once: true });

                // Connect to room
                await room.connect(url, token);
                
                // Enable microphone
                await room.localParticipant.setMicrophoneEnabled(true);
                
                updateStatus('connected', 'Connected');
                
                // Update UI
                elements.roomNameDisplay.textContent = roomName;
                elements.userNameDisplay.textContent = userName;
                elements.userIdentityDisplay.textContent = room.localParticipant.identity;
                elements.connectBtn.textContent = 'Connected';
                elements.disconnectBtn.style.display = 'block';
                
                hideModal();

                // Start audio level monitoring
                startAudioLevelMonitoring();

                console.log('Connected to room, transcription events registered');

            } catch (error) {
                console.error('Connection error:', error);
                alert(`Connection failed: ${error.message}`);
                elements.confirmConnectBtn.disabled = false;
                elements.connectBtn.disabled = false;
                updateStatus('disconnected', 'Disconnected');
            }
        }

        function handleTrackSubscribed(track, publication, participant) {
            console.log('Track subscribed:', track.kind, 'from', participant.identity);
            
            if (track.kind === 'video') {
                const element = track.attach();
                element.muted = false;
                element.volume = 1.0;
                
                element.play().catch(err => {
                    console.error('Error playing video:', err);
                });
                
                elements.remoteVideo.replaceWith(element);
                elements.remoteVideo = element;
                elements.remoteVideo.id = 'remoteVideo';
                elements.noVideoMessage.style.display = 'none';
                console.log('Video track attached');
            }
            
            if (track.kind === 'audio') {
                const element = track.attach();
                element.volume = 1.0;
                element.autoplay = true;
                element.play().catch(err => {
                    console.error('Error playing audio:', err);
                });
                elements.audioContainer.appendChild(element);
                console.log('Audio track attached');
            }
        }

        function handleTrackUnsubscribed(track, publication, participant) {
            console.log('Track unsubscribed:', track.kind);
            track.detach();
            if (track.kind === 'video') {
                elements.noVideoMessage.style.display = 'block';
            }
        }

        function handleLocalTrackPublished(publication, participant) {
            console.log('Local track published:', publication.kind);
            if (publication.kind === 'audio') {
                audioTrack = publication.track;
            }
        }

        function handleParticipantConnected(participant) {
            console.log('Participant connected:', participant.identity);
            if (participant.identity.includes('agent') || participant.identity.includes('Obama') || participant.identity.startsWith('agent-')) {
                elements.agentNameDisplay.textContent = participant.name || 'AI Agent';
                elements.agentIdentityDisplay.textContent = participant.identity;
            }
        }

        function handleDisconnect() {
            console.log('Disconnected from room');
            updateStatus('disconnected', 'Disconnected');
            resetUI();
        }

        async function disconnect() {
            if (room) {
                await room.disconnect();
                room = null;
            }
            resetUI();
        }

        function resetUI() {
            elements.roomNameDisplay.textContent = '-';
            elements.userNameDisplay.textContent = '-';
            elements.userIdentityDisplay.textContent = '-';
            elements.agentNameDisplay.textContent = 'None';
            elements.agentIdentityDisplay.textContent = 'No agent connected';
            elements.connectBtn.textContent = 'Connect';
            elements.connectBtn.disabled = false;
            elements.disconnectBtn.style.display = 'none';
            elements.noVideoMessage.style.display = 'block';
            isMuted = false;
            elements.muteBtn.classList.remove('muted');
            elements.muteBtn.innerHTML = '<span>🎤</span><span>Mute</span>';
            
            // Clear transcripts
            elements.chatMessages.innerHTML = '<div class="chat-placeholder">Transcriptions will appear here</div>';
            currentUserTranscript = null;
            currentAgentTranscript = null;
        }

        async function toggleMute() {
            if (!room) return;

            isMuted = !isMuted;
            await room.localParticipant.setMicrophoneEnabled(!isMuted);

            if (isMuted) {
                elements.muteBtn.classList.add('muted');
                elements.muteBtn.innerHTML = '<span>🔇</span><span>Unmute</span>';
            } else {
                elements.muteBtn.classList.remove('muted');
                elements.muteBtn.innerHTML = '<span>🎤</span><span>Mute</span>';
            }
        }

        function startAudioLevelMonitoring() {
            setInterval(() => {
                if (room && audioTrack && !isMuted) {
                    const level = Math.random() * 100;
                    elements.audioLevelBar.style.width = `${level}%`;
                } else {
                    elements.audioLevelBar.style.width = '0%';
                }
            }, 100);
        }

        // Event listeners
        elements.connectBtn.addEventListener('click', showModal);
        elements.confirmConnectBtn.addEventListener('click', connect);
        elements.cancelBtn.addEventListener('click', hideModal);
        elements.disconnectBtn.addEventListener('click', disconnect);
        elements.muteBtn.addEventListener('click', toggleMute);

        // Allow Enter key to connect
        elements.userName.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') connect();
        });
    </script>
</body>
</html>
    """


@app.route("/token", methods=["POST"])
def generate_token():
    """Generate a LiveKit access token"""
    try:
        data = request.get_json()
        room_name = data.get("room_name", "my-room")
        participant_name = data.get("participant_name", "user")

        # Create access token
        token = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        token.with_identity(participant_name)
        token.with_name(participant_name)
        token.with_grants(
            VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )

        jwt_token = token.to_jwt()

        return jsonify({
            "token": jwt_token,
            "url": LIVEKIT_URL,
            "room_name": room_name,
            "participant_name": participant_name
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 LiveKit Token Server Starting...")
    print("=" * 60)
    print(f"📡 Server URL: http://localhost:5000")
    print(f"🔗 LiveKit URL: {LIVEKIT_URL}")
    print(f"🎯 Open http://localhost:5000 in your browser")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=5000, debug=True)