import { auth, db, googleProvider } from "./firebase-config.js";
import {
    doc,
    setDoc,
    getDoc
} from "https://www.gstatic.com/firebasejs/12.1.0/firebase-firestore.js";

import {
    signInWithEmailAndPassword,
    createUserWithEmailAndPassword,
    sendPasswordResetEmail,
    signInWithPopup,
    updateProfile,
    sendEmailVerification,
    onAuthStateChanged,
    setPersistence,
    browserLocalPersistence,
    browserSessionPersistence
} from "https://www.gstatic.com/firebasejs/12.1.0/firebase-auth.js";

/* ---------------- Elements ---------------- */

const loginForm = document.getElementById("loginForm");
const signupForm = document.getElementById("signupForm");

const loginTab = document.getElementById("loginTab");
const signupTab = document.getElementById("signupTab");

const authMessage = document.getElementById("authMessage");

const googleBtn = document.getElementById("googleLogin");
const forgotBtn = document.getElementById("forgotPassword");

const rememberMe = document.getElementById("rememberMe");

/* ---------------- Helper ---------------- */

function showMessage(message, success = false) {

    authMessage.style.display = "block";
    authMessage.innerText = message;

    authMessage.style.background = success
        ? "#14532d"
        : "#7f1d1d";

    authMessage.style.color = "#fff";
}

/* ---------------- Tabs ---------------- */

loginTab.onclick = () => {

    loginTab.classList.add("active");
    signupTab.classList.remove("active");

    loginForm.style.display = "block";
    signupForm.style.display = "none";

};

signupTab.onclick = () => {

    signupTab.classList.add("active");
    loginTab.classList.remove("active");

    signupForm.style.display = "block";
    loginForm.style.display = "none";

};

/* ---------------- Show Password ---------------- */

document.querySelectorAll(".togglePassword").forEach(btn => {

    btn.onclick = () => {

        const input = document.getElementById(btn.dataset.target);

        input.type =
            input.type === "password"
                ? "text"
                : "password";

    };

});

/* ---------------- Login ---------------- */

loginForm.addEventListener("submit", async (e) => {

    e.preventDefault();

    try {

        await setPersistence(
            auth,
            rememberMe.checked
                ? browserLocalPersistence
                : browserSessionPersistence
        );

        const email =
            document.getElementById("loginEmail").value;

        const password =
            document.getElementById("loginPassword").value;

        const result =
            await signInWithEmailAndPassword(
                auth,
                email,
                password
            );
        if (!result.user.emailVerified) {

            showMessage("Please verify your email before logging in.");

            return;

        }
        const userDoc = await getDoc(doc(db, "users", result.user.uid));

        if (!userDoc.exists()) {

            showMessage("User profile not found.");

            return;

        }

        const role = userDoc.data().role;

        if (role === "admin") {

            window.location.href = "admin.html";

        }
        else if (role === "trainer") {

            window.location.href = "training.html";

        }
        else {

            window.location.href = "index.html";

        }

    } catch (err) {

        showMessage(err.message);

    }

});

/* ---------------- Sign Up ---------------- */

signupForm.addEventListener("submit", async (e) => {

    e.preventDefault();

    try {

        const name =
            document.getElementById("signupName").value;

        const email =
            document.getElementById("signupEmail").value;

        const password =
            document.getElementById("signupPassword").value;

        const result =
            await createUserWithEmailAndPassword(
                auth,
                email,
                password
            );


        await updateProfile(result.user, {

            displayName: name

        });

        await setDoc(
            doc(db, "users", result.user.uid),

            {
                name: name,
                email: email,
                role: "user"
            }
        );

        await sendEmailVerification(result.user);

        showMessage(
            "Account created successfully. Please verify your email.",
            true
        );

    } catch (err) {

        showMessage(err.message);

    }

});

/* ---------------- Forgot Password ---------------- */

forgotBtn.onclick = async (e) => {

    e.preventDefault();

    const email =
        document.getElementById("loginEmail").value;

    if (!email) {

        showMessage("Enter your email first.");

        return;

    }

    try {

        await sendPasswordResetEmail(auth, email);

        showMessage(
            "Password reset email sent.",
            true
        );

    } catch (err) {

        showMessage(err.message);

    }

};

/* ---------------- Google Login ---------------- */

googleBtn.onclick = async () => {

    try {

        await signInWithPopup(
            auth,
            googleProvider
        );

        window.location.href = "index.html";

    } catch (err) {

        showMessage(err.message);

    }

};

/* ---------------- Already Logged In ---------------- */
/*onAuthStateChanged(auth, async (user) => {

    if (!user) return;

    try {

        const userDoc = await getDoc(doc(db, "users", user.uid));

        if (!userDoc.exists()) {
            return;
        }

        const role = userDoc.data().role;

        if (role === "admin") {

            window.location.href = "admin.html";

        }
        else if (role === "trainer") {

            window.location.href = "training.html";

        }
        else {

            window.location.href = "user.html";

      }

    } catch (error) {

        console.error(error);

    }

});*/




