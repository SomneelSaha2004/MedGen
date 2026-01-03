# MedGen Frontend

Modern React frontend for the MedGen synthetic medical data generation platform.

## 🛠️ Tech Stack

- **React 19** - UI framework
- **Material-UI v7** - Component library
- **Recharts** - Data visualization
- **Framer Motion** - Animations
- **React Router v7** - Navigation
- **Axios** - HTTP client

## 🚀 Quick Start

### Prerequisites

- Node.js 18+
- npm or yarn
- Backend server running on port 5000

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm start
```

The app will be available at [http://localhost:3000](http://localhost:3000).

## 📁 Project Structure

```
src/
├── components/          # React components
│   ├── Home.js          # Landing page
│   ├── DatasetManager.js # Dataset management hub
│   ├── DataExplorer.js  # CSV upload & preview
│   ├── DataGeneration.js # Synthetic data generation
│   ├── Analysis.js      # Data visualization
│   ├── Database.js      # RAG database management
│   ├── QueryInterface.js # Natural language queries
│   ├── Sidebar.js       # Navigation sidebar
│   ├── About.js         # About page
│   └── Acknowledgements.js # Credits & acknowledgements
├── services/
│   └── api.js           # Backend API client
├── App.js               # Main application
└── index.js             # Entry point
```

## 🎨 Components

| Component | Description |
|-----------|-------------|
| **Home** | Landing page with project overview |
| **DatasetManager** | Central hub for managing all datasets |
| **DataExplorer** | Upload & preview CSV datasets |
| **DataGeneration** | Configure and generate synthetic data |
| **Analysis** | Statistical analysis and visualizations |
| **Database** | RAG database status and queries |
| **QueryInterface** | Natural language data queries |
| **Sidebar** | Collapsible navigation menu |

## 🔧 Available Scripts

| Command | Description |
|---------|-------------|
| `npm start` | Start development server |
| `npm run build` | Build for production |
| `npm test` | Run tests |
| `npm run test:coverage` | Run tests with coverage |
| `npm run lint` | Lint source files |
| `npm run format` | Format code with Prettier |

## 🌐 API Proxy

The development server proxies API requests to `http://localhost:5000`. This is configured in `package.json`:

```json
{
  "proxy": "http://localhost:5000"
}
```

## 📦 Production Build

```bash
npm run build
```

The optimized build will be in the `build/` directory, ready for deployment.

## 🐳 Docker

```bash
# Build image
docker build -t medgen-frontend .

# Run container
docker run -p 3000:80 medgen-frontend
```

## 📖 More Information

See the main [README](../README.md) for full project documentation.

