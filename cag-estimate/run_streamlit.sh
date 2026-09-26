#!/bin/bash
# Run Streamlit Chat App

# Kill any existing streamlit processes
pkill -9 -f "streamlit run" 2>/dev/null || true
sleep 2

# Run streamlit with telemetry disabled (avoids email prompt)
export STREAMLIT_CLIENT_TELEMETRY_OPTOUT=true

.venv/bin/python -m streamlit run src/ui/streamlit_app.py \
  --logger.level=error \
  --client.showErrorDetails=false \
  --server.headless=true &

sleep 8

echo ""
echo "✅ Streamlit Chat App is running!"
echo ""
echo "🌐 Open in your browser:"
echo "   http://localhost:8501"
echo ""
echo "📝 Feature 1: Chat Interface is LIVE"
echo ""
echo "💡 Tips:"
echo "   - Type a project description"
echo "   - Click 'Send Message'"
echo "   - See messages appear in chat history"
echo "   - Use 'New Chat' to reset conversation"
echo ""
