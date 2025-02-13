import time
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

from streamlit_extras.grid import grid
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.stoggle import stoggle
import pyperclip

from utils.session_state import ChatSessionState
from config import (
    EXPORT_FORMATS,
    MESSAGE_TYPES
)

class HistoryAnalytics:
    def __init__(self):
        self.session_state = ChatSessionState()

    def render(self):
        """Render the history and analytics view."""
        colored_header(
            label="Chat History & Analytics",
            description="View your conversation history and insights",
            color_name="red-70"
        )

        st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

        col1, col2 = st.columns([3, 2])

        with col1:
            self._render_chat_history()

        with col2:
            self._render_analytics()
            st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
            self._render_export_options()

    def _render_chat_history(self):
        """Render the chat history section."""
        st.markdown("### 📜 Conversation History")
        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

        search_term = st.text_input("🔍 Search Messages", key="search_messages")
        
        cols = st.columns([1, 1])
        with cols[0]:
            start_date = st.date_input(
                "From Date",
                value=datetime.now().date() - timedelta(days=7)
            )
        with cols[1]:
            end_date = st.date_input(
                "To Date",
                value=datetime.now().date()
            )

        messages = self._get_filtered_messages(search_term, start_date, end_date)

        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

        if not messages:
            st.info("No messages found for the selected criteria.")
            return

        for message in messages:
            self._render_message(message)

    def _render_message(self, message: Dict[str, Any]):
        """Render a single message with proper styling."""
        is_user = message['role'] == MESSAGE_TYPES["USER"]
        bg_color = "rgba(0, 200, 255, 0.1)" if is_user else "rgba(0, 78, 146, 0.1)"
        border_color = "#00C8FF" if is_user else "#00C8FF"
    
        with stylable_container(
            key=f"message_container_{message['timestamp']}",
            css_styles=f"""
                {{
                    background: {bg_color};
                    border-radius: 0.5rem;
                    padding: 1rem;
                    margin-bottom: 1rem;
                    border-left: 3px solid {border_color};
                }}
            """
        ):
            try:
                timestamp = datetime.fromisoformat(message['timestamp'])
                formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                formatted_timestamp = "Unknown time"
    
            role_display = "👤 User" if is_user else "🤖 Assistant"
            st.markdown(
                f"**{role_display}** - {formatted_timestamp}",
                help="Message sent at " + formatted_timestamp
            )
    
            message_content = message['content']
            st.markdown(message_content)
    
            cols = st.columns([1, 1, 1])
    
            with cols[0]:
                if st.button("📋 Copy", key=f"copy_{message['timestamp']}"):
                    try:
                        pyperclip.copy(message_content)
                        st.success("Copied to clipboard!")
                    except Exception as e:
                        st.error(f"Failed to copy: {str(e)}")
    
            with cols[1]:
                if st.button("🔄 Retry", key=f"retry_{message['timestamp']}",
                             disabled=not is_user):
                    self._retry_message(message)
    
            with cols[2]:
                if st.button("❌ Delete", key=f"delete_{message['timestamp']}"):
                    self.session_state.message_manager.delete_message(message['timestamp'])
                    st.rerun()
    
            if message.get('metadata'):
                with st.expander("📎 Metadata"):
                    st.json(message['metadata'])

    def _get_filtered_messages(
    self,
    search_term: str,
    start_date: datetime,
    end_date: datetime
) -> List[Dict[str, Any]]:
        """Get filtered messages using SessionStateManager."""
        # Get all messages from session state
        all_messages = self.session_state.get_messages()
        
        filtered_messages = []
        for msg in all_messages:
            try:
                # Skip system messages and None values
                if not msg or msg.get('role') == MESSAGE_TYPES["SYSTEM"]:
                    continue

                # Convert timestamp to datetime for comparison
                msg_date = datetime.fromisoformat(msg['timestamp']).date()
                
                # Check if message is within date range and matches search term
                if (start_date <= msg_date <= end_date and 
                    (not search_term or search_term.lower() in msg.get('content', '').lower())):
                    # Ensure all required fields are present
                    if all(key in msg for key in ['role', 'content', 'timestamp']):
                        filtered_messages.append(msg)
                
            except Exception as e:
                print(f"Error processing message for filtering: {str(e)}")
                continue
        
        # Sort messages by timestamp to maintain chronological order
        return sorted(filtered_messages, key=lambda x: x['timestamp'])

    def _calculate_analytics(self) -> Dict[str, Any]:
        """Calculate analytics data using SessionStateManager."""
        session_info = self.session_state.get_session_info()
        messages = self.session_state.get_messages()
        
        # Filter out system messages
        messages = [m for m in messages if m['role'] != MESSAGE_TYPES["SYSTEM"]]
        
        # Calculate response times
        response_times = []
        for i in range(len(messages)-1):
            if (messages[i]['role'] == MESSAGE_TYPES["USER"] and 
                messages[i+1]['role'] == MESSAGE_TYPES["ASSISTANT"]):
                try:
                    time_diff = datetime.fromisoformat(messages[i+1]['timestamp']) - \
                            datetime.fromisoformat(messages[i]['timestamp'])
                    response_times.append(time_diff.total_seconds())
                except Exception as e:
                    print(f"Error calculating response time: {str(e)}")
                    continue
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Message distribution
        message_distribution = {
            'User': session_info['user_messages'],
            'Assistant': session_info['assistant_messages']
        }
        
        # Activity timeline
        timeline_data = {}
        for msg in messages:
            try:
                date = datetime.fromisoformat(msg['timestamp']).date()
                if date not in timeline_data:
                    timeline_data[date] = {'total': 0, 'user': 0, 'assistant': 0}
                timeline_data[date]['total'] += 1
                if msg['role'] == MESSAGE_TYPES["USER"]:
                    timeline_data[date]['user'] += 1
                elif msg['role'] == MESSAGE_TYPES["ASSISTANT"]:
                    timeline_data[date]['assistant'] += 1
            except Exception as e:
                print(f"Error processing timeline data: {str(e)}")
                continue

        # Calculate trends
        message_trend = 0
        response_time_trend = 0
        if messages:
            try:
                recent_date = datetime.fromisoformat(messages[-1]['timestamp']).date()
                old_messages = [m for m in messages if datetime.fromisoformat(m['timestamp']).date() < recent_date]
                message_trend = len(messages) - len(old_messages)
                
                if len(response_times) > 1:
                    response_time_trend = response_times[-1] - sum(response_times[:-1]) / (len(response_times) - 1)
            except Exception as e:
                print(f"Error calculating trends: {str(e)}")
        
        return {
            'total_messages': len(messages),
            'user_messages': session_info['user_messages'],
            'assistant_messages': session_info['assistant_messages'],
            'message_trend': message_trend,
            'avg_response_time': avg_response_time,
            'response_time_trend': response_time_trend,
            'message_distribution': message_distribution,
            'activity_timeline': timeline_data,
        }

    def _render_analytics(self):
        """Render analytics and insights."""
        st.markdown("### 📊 Analytics")
        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

        analytics = self._calculate_analytics()

        # Create metrics grid with better spacing
        metrics_grid = grid(2, vertical_align="center")
        
        with metrics_grid.container():
            st.metric(
                "Total Messages",
                analytics['total_messages'],
                delta=analytics['message_trend'],
                help="Total number of messages in conversation"
            )
        
        with metrics_grid.container():
            st.metric(
                "Avg Response Time",
                f"{analytics['avg_response_time']:.1f}s",
                delta=analytics['response_time_trend'],
                help="Average time between user message and assistant response"
            )

        # Second row of metrics
        metrics_grid = grid(2, vertical_align="center")
        
        with metrics_grid.container():
            st.metric(
                "User Messages",
                analytics['user_messages'],
                help="Number of messages sent by user"
            )
        
        with metrics_grid.container():
            st.metric(
                "Assistant Messages",
                analytics['assistant_messages'],
                help="Number of messages sent by assistant"
            )

        style_metric_cards()

        st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

        # Message distribution chart
        st.plotly_chart(
            self._create_message_distribution_chart(analytics['message_distribution']),
            use_container_width=True,
            config={'displayModeBar': False}
        )

        # Activity timeline
        st.plotly_chart(
            self._create_activity_timeline(analytics['activity_timeline']),
            use_container_width=True,
            config={'displayModeBar': False}
        )

    def _create_message_distribution_chart(self, distribution: Dict[str, int]) -> go.Figure:
        """Create a message distribution chart."""
        colors = {'User': '#FF4B4B', 'Assistant': '#4287f5'}
        
        fig = go.Figure(data=[
            go.Pie(
                labels=list(distribution.keys()),
                values=list(distribution.values()),
                hole=.3,
                marker_colors=[colors[k] for k in distribution.keys()]
            )
        ])
        
        fig.update_layout(
            title="Message Distribution",
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(
                size=14,
                color='white'
            ),
            margin=dict(t=40, b=20, l=20, r=20),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=12)
            ),
            height=300
        )
        
        return fig

    def _create_activity_timeline(self, timeline_data: Dict[datetime, Dict[str, int]]) -> go.Figure:
        """Create an activity timeline chart."""
        dates = sorted(timeline_data.keys())
        user_counts = [timeline_data[date]['user'] for date in dates]
        assistant_counts = [timeline_data[date]['assistant'] for date in dates]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=user_counts,
            name='User Messages',
            mode='lines+markers',
            line=dict(color='#FF4B4B', width=2),
            marker=dict(color='#FF4B4B', size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=assistant_counts,
            name='Assistant Messages',
            mode='lines+markers',
            line=dict(color='#4287f5', width=2),
            marker=dict(color='#4287f5', size=8)
        ))
        
        fig.update_layout(
            title="Activity Timeline",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(
                size=14,
                color='white'
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                title="Date",
                title_font=dict(size=12)
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                title="Messages",
                title_font=dict(size=12)
            ),
            margin=dict(t=40, b=20, l=40, r=20),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=12)
            ),
            height=300
        )
        
        return fig

    def _retry_message(self, message: Dict[str, Any]):
        """Retry a specific message."""
        try:
            if message['role'] == MESSAGE_TYPES["USER"]:
                # Update session state to indicate a message needs processing
                if "retry_message" not in st.session_state:
                    st.session_state.retry_message = None
                
                # Store the message for processing
                st.session_state.retry_message = {
                    'content': message['content'],
                    'metadata': message.get('metadata', {}),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add success message
                st.success("Message queued for retry. Redirecting to chat...")
                time.sleep(1)  # Brief pause for user feedback
                
                # Set view to chat and rerun
                st.query_params["view"] = "chat"
                st.switch_page("main.py")
                st.rerun()

        except Exception as e:
            st.error(f"Failed to retry message: {str(e)}")

    def _export_chat_history(self, format_type: str) -> BytesIO:
        """Export chat history in the specified format."""
        messages = self.session_state.get_messages()
        messages = [m for m in messages if m['role'] != MESSAGE_TYPES["SYSTEM"]]
        
        buffer = BytesIO()
        
        if format_type == "CSV":
            df = pd.DataFrame(messages)
            df.to_csv(buffer, index=False)
        
        elif format_type == "JSON":
            df = pd.DataFrame(messages)
            df.to_json(buffer, orient='records')
        
        elif format_type == "PDF":
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            styles = getSampleStyleSheet()
            title_style = styles['Heading1']
            normal_style = styles['Normal']
            
            message_style = ParagraphStyle(
                'MessageStyle',
                parent=styles['Normal'],
                spaceBefore=10,
                spaceAfter=10,
                leftIndent=20
            )
            
            content = []
            content.append(Paragraph("Chat History Export", title_style))
            content.append(Spacer(1, 12))
            
            for message in messages:
                try:
                    timestamp = datetime.fromisoformat(message['timestamp']).strftime(
                        '%Y-%m-%d %H:%M:%S'
                    )
                    
                    role_display = "User" if message['role'] == MESSAGE_TYPES["USER"] else "Assistant"
                    message_text = f"""
                        <b>{role_display}</b> ({timestamp})<br/>
                        {message['content']}
                    """
                    content.append(Paragraph(message_text, message_style))
                    
                    if message.get('metadata'):
                        metadata_text = f"Metadata: {str(message['metadata'])}"
                        content.append(
                            Paragraph(
                                metadata_text,
                                ParagraphStyle(
                                    'MetadataStyle',
                                    parent=normal_style,
                                    leftIndent=40,
                                    fontSize=8,
                                    textColor=colors.gray
                                )
                            )
                        )
                    content.append(Spacer(1, 12))
                except Exception as e:
                    print(f"Error processing message for PDF: {str(e)}")
                    continue
            
            doc.build(content)
        
        buffer.seek(0)
        return buffer

    def _render_export_options(self):
        """Render export options."""
        st.markdown("### 📤 Export Data")
        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

        export_format = st.selectbox(
            "Export Format",
            options=list(EXPORT_FORMATS.keys()),
            help="Choose the format for exporting your chat history"
        )

        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

        if st.button("Export History", use_container_width=True):
            exported_data = self._export_chat_history(export_format)
            self._download_file(exported_data, export_format)

    def _download_file(self, buffer: BytesIO, format_type: str):
        """Create a download button for the exported file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button(
            label=f"Download {format_type}",
            data=buffer,
            file_name=f"chat_history_{timestamp}.{format_type.lower()}",
            mime=EXPORT_FORMATS[format_type]
        )