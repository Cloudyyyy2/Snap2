import { createClient, SupabaseClient } from '@supabase/supabase-js';

export interface PublicAppConfig {
  supabaseUrl: string;
  supabaseAnonKey: string;
  vapidPublicKey: string;
  groqConfigured: boolean;
  pushConfigured: boolean;
}

let cachedConfig: PublicAppConfig | null = null;
let supabaseClient: SupabaseClient | null = null;

export async function getPublicConfig(): Promise<PublicAppConfig> {
  if (cachedConfig) return cachedConfig;

  const envUrl = import.meta.env.VITE_SUPABASE_URL || '';
  const envKey = import.meta.env.VITE_SUPABASE_ANON_KEY || '';
  const envVapid = import.meta.env.VITE_VAPID_PUBLIC_KEY || '';

  try {
    const res = await fetch('/api/config');
    if (res.ok) {
      const serverConfig = await res.json();
      cachedConfig = {
        supabaseUrl: envUrl || serverConfig.supabaseUrl || '',
        supabaseAnonKey: envKey || serverConfig.supabaseAnonKey || '',
        vapidPublicKey: envVapid || serverConfig.vapidPublicKey || '',
        groqConfigured: Boolean(serverConfig.groqConfigured),
        pushConfigured: Boolean(serverConfig.pushConfigured),
      };
      return cachedConfig;
    }
  } catch (err) {
    console.warn('Could not load /api/config, falling back to Vite env:', err);
  }

  cachedConfig = {
    supabaseUrl: envUrl,
    supabaseAnonKey: envKey,
    vapidPublicKey: envVapid,
    groqConfigured: false,
    pushConfigured: Boolean(envVapid),
  };
  return cachedConfig;
}

export async function getSupabaseClient(): Promise<SupabaseClient | null> {
  if (supabaseClient) return supabaseClient;
  const config = await getPublicConfig();
  if (!config.supabaseUrl || !config.supabaseAnonKey) {
    return null;
  }
  supabaseClient = createClient(config.supabaseUrl, config.supabaseAnonKey);
  return supabaseClient;
}
