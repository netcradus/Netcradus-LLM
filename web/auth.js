import { auth } from "./firebase-config.js";

import {
    onAuthStateChanged,
    signOut
} from "https://www.gstatic.com/firebasejs/12.1.0/firebase-auth.js";

    // -----------------------------
    // Protect Page
    // -----------------------------
    onAuthStateChanged(auth, (user) => {

        // Not logged in
        if (!user) {

            if (!window.location.pathname.endsWith("login.html")) {
                window.location.href = "login.html";
            }

            return;
        }

        // -----------------------------
        // User Information
        // -----------------------------

        const name =
            user.displayName || "Netcradus User";

        const email =
            user.email || "";

        const initial =
            name.charAt(0).toUpperCase();

        // Sidebar

        setText("sidebar-user-name", name);
        setText("dropdown-user-name", name);
        setText("modal-user-name", name);

        setText("dropdown-user-email", email);
        setText("modal-user-email", email);

        setText("sidebar-user-avatar", initial);
        setText("header-avatar-trigger", initial);
        setText("modal-user-avatar", initial);

        setText(
            "sidebar-user-plan",
            "Authenticated"
        );

        setText(
            "dropdown-user-badge",
            "Authenticated"
        );

        const provider =
            user.providerData[0]?.providerId || "password";

        setText(
            "modal-user-provider",
            "Provider : " + provider
        );

    });

    // -----------------------------
    // Logout Buttons
    // -----------------------------

    const logoutButtons = [

        document.getElementById("btn-menu-logout"),

        document.getElementById("btn-modal-logout")

    ];

    logoutButtons.forEach(btn => {

        if (!btn) return;

        btn.addEventListener("click", async () => {

            try {
                localStorage.removeItem("netcradus_sessions");

                await signOut(auth);

                window.location.replace("login.html");


            }
            catch (err) {

                alert(err.message);

            }

        });

    });


// -----------------------------
// Helper
// -----------------------------

function setText(id, value) {

    const el = document.getElementById(id);

    if (el)
        el.textContent = value;

}