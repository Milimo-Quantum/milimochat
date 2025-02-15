# MilimoChat

MilimoChat is a sophisticated, locally-run chat application engineered for users who prioritize privacy, customization, and a deep understanding of their AI interactions. Built using Python and Streamlit, MilimoChat offers an intuitive interface to engage with powerful language models, manage extensive chat histories, and fine-tune every aspect of the chatbot experience. What sets MilimoChat apart is its unique memory capabilities, including both short-term and long-term memory, allowing it to provide context-aware responses and learn from interactions over time. Unlike cloud-based solutions, MilimoChat ensures your conversations and data, including its memory, remain entirely on your local machine, giving you complete control over your information. It's designed for users who want more than just a chatbot – they want a personal AI concierge that learns from their interactions, adapts to their preferences, and provides insightful analytics on their communication patterns, all within a secure, customizable environment.

## Features

- **Chat Interface**: A user-friendly chat interface for interacting with the chatbot.
- **Customization Panel**: Customize the chatbot's personality and appearance.
- **History Analytics**: Analyze chat history with visualizations and metrics.
- **Memory Dashboard**: Manage and visualize chat memory.
- **Export Service**: Export chat history and settings to various formats.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/milimo-quantum/milimochat.git
   cd milimochat
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run main.py
   ```

## Usage

### Chat Interface (components/chat_interface.py)

![Chat Interface](assets/screenshots/milimochat-1.png)

The chat interface (`ChatInterface`) is the main component for user interaction. It provides:
- **Input Area**: A text input area for users to type messages.
- **Message Display**: Displays chat messages in a conversational format, distinguishing between user and assistant messages.
- **File Upload**: Allows users to upload images and documents to provide context to the chatbot.
- **Contextual Awareness**: Integrates short-term and long-term memory to provide context-aware responses.
- **Real-time Streaming**: Displays assistant responses in real-time as they are generated.
- **Styling**: Uses custom CSS for enhanced visual appeal and user experience.

### Settings (main.py)

![Settings](assets/screenshots/milimochat-2.png)

The settings section (`main.py`) in the sidebar allows users to personalize their chat experience:
- **Model Selection**: Users can choose from a list of available language models (e.g., llama2, llama3) to power the chatbot.
  To change the default `VISION_MODEL` and `EMBED_MODEL`, modify the `config.py` file.
- **Personality Presets**: Offers a selection of personality presets (e.g., Professional, Creative, Informative) that define the chatbot's behavior and tone.
- **Tone and Creativity Control**: Sliders to adjust the tone (e.g., Formal, Balanced, Casual) and creativity level (e.g., Conservative, Balanced, Creative) of the chatbot's responses.
- **Memory Controls**: Toggles to enable or disable chat memory, allowing users to control context retention.
- **Clear Chat Button**: A button to clear the current chat history and start a new conversation.

### History Analytics (components/history_analytics.py)

![History Analytics](assets/screenshots/milimochat-3.png)

The history analytics view (`HistoryAnalytics`) provides insights into past conversations:
- **Chat History Display**: Shows a searchable and filterable history of all conversations.
- **Message Search**: Allows users to search for specific keywords or phrases within the chat history.
- **Date Filtering**: Enables users to filter messages by date range.
- **Analytics Dashboard**: Presents key metrics and visualizations, including:
    - **Total Messages**: The total number of messages exchanged.
    - **Average Response Time**: The average time taken for the assistant to respond.
    - **Message Distribution**: Charts showing the distribution of user vs. assistant messages.
    - **Activity Timeline**: Interactive timeline visualizing chat activity over time.
- **Export Options**: Allows exporting chat history in various formats (CSV, JSON, PDF).
- **Message Actions**: Options to copy, retry, or delete individual messages from the history.

### Memory Dashboard (components/memory_dashboard.py)

**Session Persistence**: MilimoChat uses session IDs to store chat memories in a local database. To ensure persistent memory across application shutdowns and restarts, it's essential to bookmark the application URL or save the session ID from the URL. This allows you to return to the same session and access your past chat history and long-term memories.


![Memory Dashboard](assets/screenshots/milimochat-4.png)

The memory dashboard (`MemoryDashboard`) offers tools to manage and understand the chatbot's memory:
- **Memory Overview**: Displays key memory metrics, such as total memories, active memories, and average memory age.
- **Memory Details**: Provides a detailed view of both short-term and long-term memory:
    - **Short-term Memory**: Shows recent messages currently in the chatbot's active memory.
    - **Long-term Memory**: Displays a searchable and paginated view of long-term memories stored in the database.
- **Memory Controls**: Includes controls for:
    - **Refresh**: Manually refresh memory data from the database.
    - **Cleanup**: Clean up old memories based on retention settings.
    - **Import/Export**: Import and export memory data to JSON files.
- **Memory Analytics**: Visualizations of memory usage and age distribution.
- **Memory Management Actions**: Options to copy, use as context, view details, and delete individual memories.

### Export Service (services/export_service.py)
The export service (`ExportService`) handles exporting chat history and settings:
- **Chat History Export**: Exports chat history to CSV, JSON, and PDF formats.
- **Settings Export**: Exports application settings to a JSON file.
- **File Download**: Provides download buttons for exported data, allowing users to save files locally.
- **Filename Generation**: Generates user-friendly filenames for exported files based on format and timestamp.
- **MIME Type Handling**: Sets appropriate MIME types for exported files to ensure correct file handling by browsers.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive messages.
4. Push your changes to your fork.
5. Create a pull request to the main repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.