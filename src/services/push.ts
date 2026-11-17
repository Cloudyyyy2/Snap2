import { getPublicConfig, getSupabaseClient } from '../lib/supabase';

function urlBase64ToUint8Array(base64String: string): Uint8Array {
  const padding = '='.repeat((4 - (base64String.length % 4)) % 4);
  const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
  const rawData = window.atob(base64);
  const outputArray = new Uint8Array(rawData.length);
  for (let i = 0; i < rawData.length; ++i) {
    outputArray[i] = rawData.charCodeAt(i);
  }
  return outputArray;
}

export function isPushSupported(): boolean {
  return (
    typeof window !== 'undefined' &&
    'serviceWorker' in navigator &&
    'PushManager' in window &&
    'Notification' in window
  );
}

export async function getExistingPushSubscription(): Promise<PushSubscription | null> {
  if (!isPushSupported()) return null;
  const registration = await navigator.serviceWorker.getRegistration('/sw.js');
  if (!registration) return null;
  return registration.pushManager.getSubscription();
}

export async function subscribeToWebPush(userId?: string): Promise<PushSubscription> {
  if (!isPushSupported()) {
    throw new Error('Web Push notifications are not supported in this browser.');
  }

  const permission = await Notification.requestPermission();
  if (permission !== 'granted') {
    throw new Error('Notification permission was denied.');
  }

  const config = await getPublicConfig();
  if (!config.vapidPublicKey) {
    throw new Error('VAPID public key is not configured.');
  }

  const registration = await navigator.serviceWorker.register('/sw.js');
  await navigator.serviceWorker.ready;

  let subscription = await registration.pushManager.getSubscription();
  if (!subscription) {
    const applicationServerKey = urlBase64ToUint8Array(config.vapidPublicKey);
    subscription = await registration.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: applicationServerKey.buffer as ArrayBuffer,
    });
  }

  const subJson = subscription.toJSON();

  // Register on server for immediate push delivery
  await fetch('/api/push/subscribe', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ subscription: subJson, userId }),
  });

  // Also persist to Supabase push_subscriptions when signed in
  if (userId) {
    const supabase = await getSupabaseClient();
    if (supabase && subJson.endpoint && subJson.keys) {
      await supabase.from('push_subscriptions').upsert(
        {
          user_id: userId,
          endpoint: subJson.endpoint,
          keys: subJson.keys,
        },
        { onConflict: 'endpoint' }
      );
    }
  }

  return subscription;
}

export async function unsubscribeFromWebPush(userId?: string): Promise<void> {
  const subscription = await getExistingPushSubscription();
  if (!subscription) return;

  const endpoint = subscription.endpoint;
  await subscription.unsubscribe();

  await fetch('/api/push/unsubscribe', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ endpoint }),
  });

  if (userId) {
    const supabase = await getSupabaseClient();
    if (supabase) {
      await supabase
        .from('push_subscriptions')
        .delete()
        .eq('user_id', userId)
        .eq('endpoint', endpoint);
    }
  }
}

export async function sendTestPushNotification(): Promise<void> {
  const subscription = await getExistingPushSubscription();
  if (!subscription) {
    throw new Error('Please enable push notifications first.');
  }

  const res = await fetch('/api/push/send-test', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ subscription: subscription.toJSON() }),
  });

  const data = await res.json();
  if (!res.ok || !data.success) {
    throw new Error(data.error || 'Failed to send test push notification.');
  }
}
