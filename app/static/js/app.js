document.addEventListener("DOMContentLoaded", () => {

    // ============================================================
    // ELEMENTS - MATCHING index.html
    // ============================================================

    const chatContainer = document.getElementById("chat-container");
    const questionInput = document.getElementById("question");
    const employeeInput = document.getElementById("employee-id");
    const sendButton = document.getElementById("send-button");
    const clearButton = document.getElementById("clear-button");

    const sourcesContainer = document.getElementById("sources-container");
    const sourcesList = document.getElementById("sources-list");


    // ============================================================
    // ESCAPE HTML
    // ============================================================

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text ?? "";
        return div.innerHTML;
    }


    // ============================================================
    // FORMAT ASSISTANT RESPONSE
    // ============================================================

    function formatAnswer(text) {

        if (!text) {
            return "";
        }

        let html = escapeHtml(text);

        // Bold
        html = html.replace(
            /\*\*(.*?)\*\*/g,
            "<strong>$1</strong>"
        );

        // Inline code
        html = html.replace(
            /`([^`]+)`/g,
            "<code>$1</code>"
        );

        // Line breaks
        html = html.replace(/\n/g, "<br>");

        return html;
    }


    // ============================================================
    // ADD USER MESSAGE
    // ============================================================

    function addUserMessage(text) {

        const message = document.createElement("div");

        message.className = "message user-message";

        message.innerHTML = `
            <div class="message-label">
                👤 You
            </div>

            <div class="message-text">
                ${escapeHtml(text)}
            </div>
        `;

        chatContainer.appendChild(message);

        scrollToBottom();
    }


    // ============================================================
    // ADD ASSISTANT MESSAGE
    // ============================================================

    function addAssistantMessage(text) {

        const message = document.createElement("div");

        message.className = "message assistant-message";

        message.innerHTML = `
            <div class="message-label">
                🤖 Assistant
            </div>

            <div class="message-text">
                ${formatAnswer(text)}
            </div>
        `;

        chatContainer.appendChild(message);

        scrollToBottom();
    }


    // ============================================================
    // LOADING MESSAGE
    // ============================================================

    function showLoading() {

        removeLoading();

        const message = document.createElement("div");

        message.id = "loading-message";
        message.className = "message assistant-message";

        message.innerHTML = `
            <div class="message-label">
                🤖 Assistant
            </div>

            <div class="message-text">
                Thinking...
            </div>
        `;

        chatContainer.appendChild(message);

        scrollToBottom();
    }


    function removeLoading() {

        const loading =
            document.getElementById("loading-message");

        if (loading) {
            loading.remove();
        }
    }


    // ============================================================
    // SHOW POLICY SOURCES
    // ============================================================

    function showSources(sources) {

        if (!sourcesContainer || !sourcesList) {
            return;
        }

        sourcesList.innerHTML = "";

        if (!sources || sources.length === 0) {

            sourcesContainer.classList.add("hidden");

            return;
        }


        sources.forEach((source) => {

            let sourceName = "";
            let policyType = "policy";
            let country = "All";


            // ----------------------------------------------------
            // CURRENT BACKEND FORMAT
            //
            // [
            //     "airport_policy.txt",
            //     "approval_policy.txt"
            // ]
            // ----------------------------------------------------

            if (typeof source === "string") {

                sourceName = source;


                if (source === "travel_policy_india.txt") {

                    policyType = "travel";
                    country = "India";

                }

                else if (source === "travel_policy_us.txt") {

                    policyType = "travel";
                    country = "United States";

                }

                else if (source === "airport_policy.txt") {

                    policyType = "airport";

                }

                else if (source === "employee_eligibility.txt") {

                    policyType = "eligibility";

                }

                else if (source === "expense_policy.txt") {

                    policyType = "expense";

                }

                else if (source === "cancellation_policy.txt") {

                    policyType = "cancellation";

                }

                else if (source === "approval_policy.txt") {

                    policyType = "approval";

                }

            }


            // ----------------------------------------------------
            // FUTURE OBJECT FORMAT
            // ----------------------------------------------------

            else if (
                typeof source === "object" &&
                source !== null
            ) {

                const metadata =
                    source.metadata || {};

                sourceName =
                    metadata.source ||
                    source.source ||
                    source.file ||
                    "Unknown policy";

                policyType =
                    metadata.policy_type ||
                    source.policy_type ||
                    "policy";

                country =
                    metadata.country ||
                    source.country ||
                    "All";
            }


            // ----------------------------------------------------
            // CREATE SOURCE CARD
            // ----------------------------------------------------

            const card = document.createElement("div");

            card.className = "source-card";

            card.innerHTML = `
                <div class="source-icon">
                    📄
                </div>

                <div class="source-info">

                    <div class="source-name">
                        ${escapeHtml(sourceName)}
                    </div>

                    <div class="source-meta">
                        ${escapeHtml(policyType)}
                    </div>

                    <div class="source-country">
                        🌍 ${escapeHtml(country)}
                    </div>

                </div>
            `;

            sourcesList.appendChild(card);
        });


        sourcesContainer.classList.remove("hidden");
    }


    // ============================================================
    // SEND QUESTION
    // ============================================================

    async function sendQuestion() {

        const question =
            questionInput.value.trim();

        const employeeId =
            employeeInput
                ? employeeInput.value.trim()
                : "";


        // --------------------------------------------------------
        // EMPTY QUESTION
        // --------------------------------------------------------

        if (!question) {

            addAssistantMessage(
                "Please enter a question."
            );

            return;
        }


        // --------------------------------------------------------
        // SHOW USER QUESTION IMMEDIATELY
        // --------------------------------------------------------

        addUserMessage(question);


        // Clear input AFTER capturing the question
        questionInput.value = "";


        // Disable button while processing
        sendButton.disabled = true;


        // Show loading
        showLoading();


        // --------------------------------------------------------
        // CALL FLASK
        // --------------------------------------------------------

        try {

            const response = await fetch("/ask", {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({

                    employee_id: employeeId,

                    question: question

                })

            });


            const data =
                await response.json();


            removeLoading();


            // ----------------------------------------------------
            // SERVER ERROR
            // ----------------------------------------------------

            if (!response.ok || !data.success) {

                addAssistantMessage(
                    data.error ||
                    "Sorry, something went wrong while processing your question."
                );

                return;
            }


            // ----------------------------------------------------
            // SUCCESS
            // ----------------------------------------------------

            addAssistantMessage(
                data.answer
            );


            showSources(
                data.sources || []
            );

        }


        catch (error) {

            removeLoading();

            console.error(
                "Request error:",
                error
            );

            addAssistantMessage(
                "Unable to connect to the Travel Policy Assistant."
            );

        }


        finally {

            sendButton.disabled = false;

            questionInput.focus();

        }

    }


    // ============================================================
    // CLEAR CONVERSATION
    // ============================================================

    async function clearConversation() {

        try {

            const response =
                await fetch("/clear", {
                    method: "POST"
                });


            const data =
                await response.json();


            if (!response.ok || !data.success) {

                console.error(
                    "Clear error:",
                    data.error
                );

                return;
            }


            // Keep the original welcome message
            const welcome =
                chatContainer.querySelector(
                    ".assistant-message"
                );


            chatContainer.innerHTML = "";


            if (welcome) {

                chatContainer.appendChild(
                    welcome
                );

            }


            if (sourcesContainer) {

                sourcesContainer.classList.add(
                    "hidden"
                );

            }


            if (sourcesList) {

                sourcesList.innerHTML = "";

            }

        }


        catch (error) {

            console.error(
                "Clear request failed:",
                error
            );

        }

    }


    // ============================================================
    // SEND BUTTON
    // ============================================================

    if (sendButton) {

        sendButton.addEventListener(
            "click",
            sendQuestion
        );

    }


    // ============================================================
    // CLEAR BUTTON
    // ============================================================

    if (clearButton) {

        clearButton.addEventListener(
            "click",
            clearConversation
        );

    }


    // ============================================================
    // ENTER KEY
    // ============================================================

    if (questionInput) {

        questionInput.addEventListener(
            "keydown",
            (event) => {

                if (
                    event.key === "Enter" &&
                    !event.shiftKey
                ) {

                    event.preventDefault();


                    if (
                        !sendButton.disabled
                    ) {

                        sendQuestion();

                    }

                }

            }
        );

    }


    // ============================================================
    // QUICK QUESTIONS
    // ============================================================

    document
        .querySelectorAll(".quick-question")
        .forEach((button) => {

            button.addEventListener(
                "click",
                () => {

                    const question =
                        button.dataset.question ||
                        button.textContent.trim();

                    questionInput.value =
                        question;

                    questionInput.focus();

                }
            );

        });


    // ============================================================
    // SCROLL
    // ============================================================

    function scrollToBottom() {

        chatContainer.scrollTop =
            chatContainer.scrollHeight;

    }

});