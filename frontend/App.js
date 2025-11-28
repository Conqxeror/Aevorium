import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Provider as PaperProvider } from 'react-native-paper';
import OverviewScreen from './src/screens/OverviewScreen';
import GenerateScreen from './src/screens/GenerateScreen';
import PrivacyScreen from './src/screens/PrivacyScreen';
import AuditLogScreen from './src/screens/AuditLogScreen';

const Tab = createBottomTabNavigator();

export default function App() {
  return (
    <PaperProvider>
      <NavigationContainer>
        <Tab.Navigator
          initialRouteName="Overview"
          screenOptions={{ headerShown: true }}
        >
          <Tab.Screen name="Overview" component={OverviewScreen} />
          <Tab.Screen name="Generate" component={GenerateScreen} />
          <Tab.Screen name="Privacy" component={PrivacyScreen} />
          <Tab.Screen name="Audit" component={AuditLogScreen} />
        </Tab.Navigator>
      </NavigationContainer>
    </PaperProvider>
  );
}
