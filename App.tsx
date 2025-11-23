import React, { useState } from 'react';
import Navigation from './components/Navigation';
import Landing from './components/Landing';
import Dashboard from './components/Dashboard';
import AnalysisTool from './components/AnalysisTool';
import Footer from './components/Footer';
import Team from './components/Team';

const App: React.FC = () => {
  const [currentView, setCurrentView] = useState('home');

  const renderView = () => {
    switch (currentView) {
      case 'home':
        return <Landing onGetStarted={() => setCurrentView('dashboard')} />;
      case 'dashboard':
        return <Dashboard />;
      case 'analysis':
        return <AnalysisTool />;
      case 'team':
        return <Team />;
      default:
        return <Landing onGetStarted={() => setCurrentView('dashboard')} />;
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col font-sans">
      <Navigation currentView={currentView} setView={setCurrentView} />
      <main className="flex-grow">
        {renderView()}
      </main>
      <Footer />
    </div>
  );
};

export default App;