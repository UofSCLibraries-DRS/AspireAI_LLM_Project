import { useRef } from 'react';

export function useChatRefs() {
    return {
        chatLog: useRef<HTMLDivElement | null>(null),
        welcome: useRef<HTMLDivElement | null>(null),
        userInput: useRef<HTMLTextAreaElement | null>(null),
        sendBtn: useRef<HTMLButtonElement | null>(null),
        modelSelector: useRef<HTMLSelectElement | null>(null),
    };
}
