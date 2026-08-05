// Firebase Configuration
import { initializeApp } from "https://www.gstatic.com/firebasejs/12.1.0/firebase-app.js";

import {
    getAuth,
    GoogleAuthProvider
} from "https://www.gstatic.com/firebasejs/12.1.0/firebase-auth.js";

// =============================
const firebaseConfig = {
    apiKey: "AIzaSyD64Q5Xkb6mIxQPMAQAF4Vv0XfCRrbaAu0",
    authDomain: "netcradus-llm.firebaseapp.com",
    projectId: "netcradus-llm",
    storageBucket: "netcradus-llm.firebasestorage.app",
    messagingSenderId: "919864881482",
    appId: "1:919864881482:web:91cc287020214b3b4fd1b3"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Authentication
const auth = getAuth(app);

// Google Provider
const googleProvider = new GoogleAuthProvider();

export {
    auth,
    googleProvider
};