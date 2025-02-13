import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from config import EXPORT_FORMATS

class ExportService:
    def __init__(self):
        self.supported_formats = EXPORT_FORMATS

    def export_chat_history(
        self,
        messages: List[Dict[str, Any]],
        format_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BytesIO:
        """Export chat history in the specified format."""
        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format_type}")

        buffer = BytesIO()
        
        try:
            if format_type == "CSV":
                self._export_to_csv(messages, buffer)
            elif format_type == "JSON":
                self._export_to_json(messages, buffer, metadata)
            elif format_type == "PDF":
                self._export_to_pdf(messages, buffer, metadata)
            
            buffer.seek(0)
            return buffer
        
        except Exception as e:
            raise Exception(f"Failed to export chat history: {str(e)}")

    def _export_to_csv(self, messages: List[Dict[str, Any]], buffer: BytesIO):
        """Export chat history to CSV format."""
        df = pd.DataFrame(messages)
        
        # Format timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Handle metadata column
        if 'metadata' in df.columns:
            df['metadata'] = df['metadata'].apply(lambda x: json.dumps(x) if x else '')
        
        df.to_csv(buffer, index=False)

    def _export_to_json(
        self,
        messages: List[Dict[str, Any]],
        buffer: BytesIO,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Export chat history to JSON format."""
        export_data = {
            'metadata': metadata or {
                'export_date': datetime.now().isoformat(),
                'message_count': len(messages)
            },
            'messages': messages
        }
        
        json.dump(
            export_data,
            buffer,
            indent=2,
            default=str  # Handle datetime serialization
        )

    def _export_to_pdf(
        self,
        messages: List[Dict[str, Any]],
        buffer: BytesIO,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Export chat history to PDF format."""
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        # Create styles
        styles = getSampleStyleSheet()
        title_style = styles['Heading1']
        normal_style = styles['Normal']
        
        # Custom style for messages
        message_style = ParagraphStyle(
            'MessageStyle',
            parent=styles['Normal'],
            spaceBefore=10,
            spaceAfter=10,
            leftIndent=20
        )

        # Build content
        content = []
        
        # Add title
        content.append(Paragraph("Chat History Export", title_style))
        content.append(Spacer(1, 12))
        
        # Add metadata
        if metadata:
            content.append(Paragraph("Export Information:", styles['Heading2']))
            for key, value in metadata.items():
                content.append(
                    Paragraph(f"{key}: {value}", normal_style)
                )
            content.append(Spacer(1, 12))

        # Add messages
        content.append(Paragraph("Messages:", styles['Heading2']))
        content.append(Spacer(1, 12))
        
        for message in messages:
            # Format timestamp
            timestamp = datetime.fromisoformat(message['timestamp']).strftime(
                '%Y-%m-%d %H:%M:%S'
            )
            
            # Create message block
            message_text = f"""
                <b>{message['role'].title()}</b> ({timestamp})<br/>
                {message['content']}
            """
            
            content.append(Paragraph(message_text, message_style))
            
            # Add metadata if present
            if message.get('metadata'):
                metadata_text = f"Metadata: {json.dumps(message['metadata'], indent=2)}"
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

        # Build PDF
        doc.build(content)

    def get_export_filename(self, format_type: str) -> str:
        """Generate filename for export."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"chat_history_{timestamp}.{format_type.lower()}"

    def get_mime_type(self, format_type: str) -> str:
        """Get MIME type for export format."""
        mime_types = {
            "CSV": "text/csv",
            "JSON": "application/json",
            "PDF": "application/pdf"
        }
        return mime_types.get(format_type, "application/octet-stream")